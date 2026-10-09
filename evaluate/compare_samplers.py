"""Direct comparison of the two samplers, Metropolis and Gibbs, for several N.

For every N:
1. Tune the step_size of both samplers so that their acceptance is inside [ACC_MIN, ACC_MAX] of train.py,
   first at the initial theta and again at a trained reference theta (the acceptance drops while psi narrows).
2. Train from the same initial theta with both samplers and record the energy against iterations and wall time.
3. Sample the reference theta with both samplers, recording every step, and measure whether they agree, the
   autocorrelation time per step and the effective samples per second.

Results are saved in evaluate/output/compare_N{...}_{date-time}/ and analysed in section 7 of
notebooks/analysis.ipynb. Run from the project root:
    uv run python evaluate/compare_samplers.py --N 4 20
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from pathlib import Path

#The script lives in evaluate/, but src/ and train.py are in the project root (where pyproject.toml is)
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

#External libraries
import jax
import jax.numpy as jnp
import numpy as np
import argparse
import tomllib
import time
import copy
import json
import shutil
from datetime import datetime

#Project modules
from src.boson_trap import boson_trap
from src.nqs import NQS
from src.sampler_m import MetroSampler
from src.sampler_g import GibbsSampler
from src.sr_optimizer import SR, flatten_params
from train import make_train_step, SAMPLER_TAGS, ACC_MIN, ACC_MAX


def autocorr_time(x):
    """Integrated autocorrelation time tau = 1 + 2 sum_k rho(k) of x with shape (n_times, n_chains),
    in units of the time step of x"""
    #First we need to make sure we are using numpy arrays

    x=np.asarray(x)

    x_centered = x -x.mean(axis=0) #(n_sweeps, n_chains)- (n_chains,)
    #Variance C(0): mean of the squared fluctuations over all sweeps and chains. It normalizes rho(0) = 1
    c0 = np.mean(x_centered**2)

    #Sum of the normalized autocorrelations rho(k) = C(k)/C(0) up to the window
    rho_sum = 0.0
    for k in range(1, x.shape[0]):
        #Every sweep t times the sweep t+k of the same chain: both slices are (n_sweeps-k, n_chains),
        #so chains are never mixed. The mean averages over the pairs (t, t+k) and over the chains
        rho = np.mean(x_centered[:-k] * x_centered[k:]) / c0

        #Window: stop at the first non-positive rho. Beyond it rho is dominated by noise and the sum
        #would not converge. This slightly underestimates tau, equally for both samplers
        if rho <= 0:
            break
        rho_sum += rho

    #Integrated autocorrelation time: n correlated samples are worth n/tau independent ones
    return 1 + 2 * rho_sum

def make_samplers(cfg, log_psi, unravel):
    """Essentially the same as sample_fn in train.py, for both samplers at once.
    Returns {"metropolis": (sampler, sample_fn), "gibbs": (sampler, sample_fn)}"""
    #Both samplers are needed, so [sampler] type is not read: each one takes its own subsection

    sampler_m = MetroSampler(**cfg["sampler"]["metropolis"])
    sampler_g = GibbsSampler(**cfg["sampler"]["gibbs"])

    #Metropolis only evaluates psi, so it takes log_psi(x) with fixed theta

    sample_m=lambda t, w, n, k: sampler_m.sample(lambda x: log_psi(t,x), w, n, k)

    #Gibbs rebuilds |psi|^2 from the encoder, the RBM and alpha, so it needs the parameters dictionary (unravel)

    sample_g=lambda t, w, n, k: sampler_g.sample(unravel(t), w, n, k)

    #Tuples (sampler, sample_fn): a set {a, b} has no order, so it could not be unpacked reliably
    return {"metropolis": (sampler_m, sample_m), "gibbs": (sampler_g, sample_g)}

def setup(cfg, N):
    """Build everything that depends on N but not on the sampler: system, model, initial theta, log_psi(theta, x), unravel and samplers"""

    #We override the n_particles value of the config
    system=boson_trap(**{**cfg["system"], "n_particles": N})

    #Model and initial parameters (as in train)

    model = NQS(**cfg["network"], dim=system.dim)  #The dimension of the particles must match the system
    params = model.init(jax.random.PRNGKey(cfg["training"]["seed"]))

    #We flatten the parameters of the model to get theta, and we also get the unravel function to go back to the original structure
    theta, unravel = flatten_params(params)

    #We also need to build log_psi(t,x)
    log_psi = lambda t, x: model.apply(unravel(t), x)

    #And lastly the samplers (with sample_fn signature)
    samplers=make_samplers(cfg, log_psi, unravel)

    return {"system": system, "theta": theta, "unravel": unravel, "log_psi": log_psi, "samplers": samplers}


def tune_step_size(cfg, S, name, theta, key, n_meas=20, max_trials=12):
    """Find a step_size of sampler `name` whose acceptance at `theta` is inside [ACC_MIN, ACC_MAX].
    Bisection in log scale: the step is doubled or halved until the band is bracketed, then the geometric
    mean of the bracket is tried. The walkers are thermalized once and carried between trials, because the
    distribution |psi|^2 does not depend on the step size.
    Returns the tuned step_size, its acceptance and the list of (step_size, acceptance) tried."""

    cfg = copy.deepcopy(cfg)
    step = cfg["sampler"][name]["step_size"]
    lo, hi = None, None   #largest step with too high acceptance / smallest step with too low acceptance
    trials = []
    key_w, key_therm, key = jax.random.split(key, 3)

    for i in range(max_trials):
        cfg["sampler"][name]["step_size"] = step
        sampler, sample_fn = make_samplers(cfg, S["log_psi"], S["unravel"])[name]
        if i == 0:
            walkers = sampler.init_walkers(key_w, S["system"].n_particles, S["system"].dim)
            _, walkers, _ = sample_fn(theta, walkers, cfg["training"]["n_thermalization"], key_therm)

        key, key_i = jax.random.split(key)
        _, walkers, acc = sample_fn(theta, walkers, n_meas, key_i)
        acc = float(acc)
        trials.append((step, acc))

        if ACC_MIN <= acc <= ACC_MAX:
            break
        if acc > ACC_MAX:
            #Too many accepted moves: the step is too small
            lo = step
            step = 2 * step if hi is None else np.sqrt(lo * hi)
        else:
            #Too many rejected moves: the step is too large
            hi = step
            step = step / 2 if lo is None else np.sqrt(lo * hi)
    else:
        print(f"  Warning: {name} step_size not tuned in {max_trials} trials, last acceptance {acc:.2f}")

    return trials[-1][0], trials[-1][1], trials


def tune_samplers(cfg, S, theta, key):
    """Tune the step_size of both samplers at theta. Returns a copy of cfg with the tuned values and a
    summary {name: {"step_size", "acceptance", "trials"}}"""
    cfg = copy.deepcopy(cfg)
    info = {}
    for name in ["metropolis", "gibbs"]:
        key, key_n = jax.random.split(key)
        step, acc, trials = tune_step_size(cfg, S, name, theta, key_n)
        cfg["sampler"][name]["step_size"] = float(step)
        info[name] = {"step_size": float(step), "acceptance": acc, "trials": trials}
        print(f"  {name:<10} step_size = {step:.4f}, acceptance = {acc:.2f} ({len(trials)} trials)")
    return cfg, info


def train_theta(cfg, S, sampler_name, n_iter, key):
    """Train from the initial theta of S with one of the two samplers, as train.py does.
    Returns the trained theta and the history (energy, variance, acceptance, wall time per iteration)."""

    sampler, sample_fn = S["samplers"][sampler_name]

    train_cfg=cfg["training"]

    key_w, key_therm, key_train = jax.random.split(key, 3)

    #Walker init and thermalization with the corresponding check. N and dim come from S["system"],
    #because setup replaced n_particles of the config
    walkers = sampler.init_walkers(key_w, S["system"].n_particles, S["system"].dim)
    therm_samples, walkers, _ = sample_fn(S["theta"], walkers, train_cfg["n_thermalization"], key_therm)
    thermalized, z = sampler.check_therm(therm_samples)
    if not thermalized:
        raise RuntimeError(f"{sampler_name}: chains have not thermalized (z = {float(z):.2f}), increase n_thermalization")

    #The same jitted SR iteration as train.py, with the sample_fn of this sampler
    train_step = make_train_step(S["log_psi"], sample_fn, S["system"], SR(**cfg["sr"]), train_cfg["n_samples"])

    theta = S["theta"]
    history = {"energy": [], "variance": [], "acceptance": [], "time": []}
    for _ in range(n_iter):
        key_train, key_iter = jax.random.split(key_train)

        #JAX runs asynchronously: block_until_ready waits for the GPU, otherwise we would only time
        #how long it takes to launch the work. The first iteration also includes the compilation
        t0 = time.perf_counter()
        theta, walkers, (energy, variance), acceptance, _ = train_step(theta, walkers, key_iter)
        jax.block_until_ready(theta)
        history["time"].append(time.perf_counter() - t0)

        history["energy"].append(float(energy))
        history["variance"].append(float(variance))
        history["acceptance"].append(float(acceptance))

    return theta, {name: np.asarray(values) for name, values in history.items()}


def mean_and_error(x):
    """Mean of x (n_times, n_chains) and its error bar from the spread of the chain means. Chains are
    independent, so this error already accounts for the autocorrelation inside each chain."""
    chain_means = x.mean(axis=0)
    return chain_means.mean(), chain_means.std(ddof=1) / np.sqrt(x.shape[1])


def compare_fixed_theta(cfg, S, theta, n_steps, key, e_stride=10):
    """Sample the same theta with both samplers, recording EVERY step (n_sweep = 1), and measure for each:
    - <sum_i |x_i|^2> and <E_loc> with error bars (test 1: both must agree within errors)
    - the integrated autocorrelation time per step and the effective samples per second (test 2)
    Recording every step makes tau independent of how much each sampler thins its chain in training.
    sum|x|^2 is cheap and is measured on every step. E_loc needs an (N*dim)^2 Hessian per sample, so it is
    measured every e_stride steps, and its tau is in units of e_stride steps.
    Returns {sampler_name: {...}} and, under "agreement", the difference between samplers in sigmas."""

    system, log_psi = S["system"], S["log_psi"]
    log_psi_x = lambda x: log_psi(theta, x)
    batch_E = jax.jit(lambda x: system.batch_local_energy(log_psi_x, x))

    #Same samplers and step sizes, but recording one configuration per step
    cfg_step = copy.deepcopy(cfg)
    for name in ["metropolis", "gibbs"]:
        cfg_step["sampler"][name]["n_sweep"] = 1
    samplers = make_samplers(cfg_step, log_psi, S["unravel"])

    results = {}
    for name, (sampler, sample_fn) in samplers.items():
        key, key_w, key_therm, key_s = jax.random.split(key, 4)
        n_chains = sampler.n_chains

        #Thermalize from scratch with as many steps as in training (n_thermalization sweeps of n_sweep steps)
        n_therm_steps = cfg["training"]["n_thermalization"] * cfg["sampler"][name]["n_sweep"]
        walkers = sampler.init_walkers(key_w, system.n_particles, system.dim)
        therm_samples, walkers, _ = sample_fn(theta, walkers, n_therm_steps, key_therm)
        thermalized, z = sampler.check_therm(therm_samples)
        if not thermalized:
            raise RuntimeError(f"{name}: chains have not thermalized (z = {float(z):.2f}), increase n_thermalization")

        #Compile first (ahead of time) and then time only the execution, waiting for the GPU
        sample = jax.jit(lambda w, k: sample_fn(theta, w, n_steps, k)).lower(walkers, key_s).compile()
        t0 = time.perf_counter()
        samples, walkers, acceptance = sample(walkers, key_s)
        jax.block_until_ready(samples)
        elapsed = time.perf_counter() - t0

        #sum|x|^2 per step, reshaped to (n_steps, n_chains): samples are step-major
        r2 = np.asarray(jnp.sum(samples**2, axis=(1, 2))).reshape(n_steps, n_chains)

        #E_loc every e_stride steps, in batches of the size of one training iteration
        sub = samples.reshape(n_steps, n_chains, *samples.shape[1:])[::e_stride].reshape(-1, *samples.shape[1:])
        batch = n_chains * cfg["training"]["n_samples"]
        E_loc = np.concatenate([np.asarray(batch_E(sub[i:i + batch])) for i in range(0, sub.shape[0], batch)])
        E_loc = E_loc.real.reshape(-1, n_chains)   #<H> is real: the imaginary part averages to 0

        res = {"step_size": cfg["sampler"][name]["step_size"], "acceptance": float(acceptance),
               "time": elapsed, "time_per_step": elapsed / n_steps}
        for obs, x, stride in [("r2", r2, 1), ("E", E_loc, e_stride)]:
            mean, err = mean_and_error(x)
            tau = autocorr_time(x)
            #x.size correlated samples are worth x.size/tau independent ones
            res[obs] = {"mean": mean, "err": err, "tau": tau, "stride": stride,
                        "ess_per_s": x.size / (tau * elapsed)}
        results[name] = res

    #Test 1: both samplers sample |psi|^2, so the means must agree within the combined error bar
    m, g = results["metropolis"], results["gibbs"]
    results["agreement"] = {obs: (m[obs]["mean"] - g[obs]["mean"]) / np.hypot(m[obs]["err"], g[obs]["err"])
                            for obs in ["r2", "E"]}
    return results


def compare_training(cfg, S, n_iter, key, rel_tol=1e-3):
    """Train from the same initial theta and with the same key with both samplers (test 3).
    For each one returns the history, the trained theta and a summary: final energy (mean of the last 10%
    of the iterations, as train.py), time per iteration, and the iterations and wall time needed to
    reach a relative error below rel_tol. The first iteration includes the compilation, so it is left
    out of the times."""

    results = {}
    exact = S["system"].exact_energy
    for name in S["samplers"]:
        #Same starting theta (S["theta"]) and same key for both, so only the sampler changes
        theta, h = train_theta(cfg, S, name, n_iter, key)

        n_last = max(1, n_iter // 10)
        rel_err = np.abs(h["energy"] - exact) / exact
        #Cumulative wall time without the compilation of the first iteration
        h["cum_time"] = np.cumsum(np.concatenate([[0.0], h["time"][1:]]))
        h["rel_err"] = rel_err
        below = np.nonzero(rel_err < rel_tol)[0]

        results[name] = {
            "theta": theta, "history": h,
            "E_final": h["energy"][-n_last:].mean(),
            "E_err": h["energy"][-n_last:].std() / np.sqrt(n_last),   #indicative, as in train.py
            "acc_final": h["acceptance"][-n_last:].mean(),
            "time_per_iter": h["time"][1:].mean(),
            "iter_tol": int(below[0]) + 1 if below.size else None,    #None: rel_tol never reached
            "time_tol": h["cum_time"][below[0]] if below.size else None,
        }
    return results


def run_N(cfg, N, args, key):
    """Whole comparison for one N: tune at the initial theta, train a reference theta with Metropolis,
    re-tune at it, compare the training from scratch and compare both samplers at the reference theta."""

    print(f"\n===== N = {N} =====")
    S = setup(cfg, N)
    k_tune0, k_ref, k_tune1, k_train, k_fixed = jax.random.split(key, 5)

    #1. Steps that work at the initial theta, needed to train the reference theta
    print("Tuning at the initial theta:")
    cfg_N, tuning_init = tune_samplers(cfg, S, S["theta"], k_tune0)
    S["samplers"] = make_samplers(cfg_N, S["log_psi"], S["unravel"])

    #2. Reference theta, then re-tune: psi narrows during training and the acceptance drops, so the
    #steps that matter are the ones for the trained psi (the same advice as for train.py)
    print(f"Training the reference theta with Metropolis ({args.n_iter} iterations)")
    theta_ref, _ = train_theta(cfg_N, S, "metropolis", args.n_iter, k_ref)
    print("Tuning at the reference theta:")
    cfg_N, tuning_ref = tune_samplers(cfg_N, S, theta_ref, k_tune1)
    S["samplers"] = make_samplers(cfg_N, S["log_psi"], S["unravel"])

    #3. Training from scratch with both samplers (test 3)
    print("Comparing the training")
    training = compare_training(cfg_N, S, args.n_iter, k_train, rel_tol=args.rel_tol)

    #4. Both samplers at the reference theta, one record per step (tests 1 and 2)
    print(f"Comparing at the reference theta ({args.n_steps} steps)")
    fixed = compare_fixed_theta(cfg_N, S, theta_ref, args.n_steps, k_fixed, e_stride=args.e_stride)

    print_summary(N, S["system"].exact_energy, training, fixed)
    return {"exact_energy": S["system"].exact_energy, "tuning_init": tuning_init, "tuning_ref": tuning_ref,
            "training": training, "fixed": fixed}


def print_summary(N, exact, training, fixed):
    print(f"\nN = {N}, E_0 = {exact:g}")
    print(f"{'sampler':<11}{'step':>8}{'acc':>6}{'E_final':>20}{'ms/iter':>9}{'it<tol':>8}{'s<tol':>8}"
          f"{'μs/step':>10}{'τ r2':>7}{'ESS/s r2':>11}{'τ E':>7}")
    for name in ["metropolis", "gibbs"]:
        t, f = training[name], fixed[name]
        E = f"{t['E_final']:.5f}±{t['E_err']:.5f}"
        time_tol = f"{t['time_tol']:.1f}" if t["time_tol"] is not None else "-"
        print(f"{name:<11}{f['step_size']:>8.4f}{f['acceptance']:>6.2f}{E:>20}{1e3 * t['time_per_iter']:>9.1f}"
              f"{str(t['iter_tol']):>8}{time_tol:>8}{1e6 * f['time_per_step']:>10.1f}{f['r2']['tau']:>7.2f}"
              f"{f['r2']['ess_per_s']:>11.3g}{f['E']['tau']:>7.2f}")
    a = fixed["agreement"]
    print(f"Agreement Metropolis - Gibbs: sum|x|^2 {a['r2']:+.2f} σ, E {a['E']:+.2f} σ")


def to_json(obj):
    """Convert the nested results to plain Python for json (numpy scalars, tuples), dropping arrays"""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items() if not isinstance(v, (np.ndarray, jax.Array))}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_results(args, all_results):
    """Save the comparison in its own folder output/compare_N{N1-N2...}_{date-time}/:
    - summary.json: arguments, tuned step sizes, training summaries and fixed-theta measurements per N
    - histories.npz: training histories, keys N{N}_{sampler}_{field}
    - config.toml: copy of the config used (before tuning; the tuned steps are in summary.json)"""

    Ns = "-".join(str(N) for N in all_results)
    run_dir = Path(args.output) / f"compare_N{Ns}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "sampler_tags": SAMPLER_TAGS, "acc_range": [ACC_MIN, ACC_MAX], "N": {}}
    histories = {}
    for N, res in all_results.items():
        summary["N"][str(N)] = to_json({k: v for k, v in res.items()})
        for name, t in res["training"].items():
            for field, values in t["history"].items():
                histories[f"N{N}_{name}_{field}"] = values

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    np.savez(run_dir / "histories.npz", **histories)
    shutil.copy(args.config, run_dir / "config.toml")
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Compare the Metropolis and Gibbs samplers for several N")
    parser.add_argument("--config", default=str(ROOT / "config.toml"), help="TOML config; n_particles is replaced by --N")
    parser.add_argument("--N", type=int, nargs="+", default=[4, 20], help="Numbers of particles to compare")
    parser.add_argument("--n_iter", type=int, default=150, help="SR iterations of every training")
    parser.add_argument("--n_steps", type=int, default=1000, help="Steps recorded per chain at the reference theta")
    parser.add_argument("--e_stride", type=int, default=10, help="E_loc is measured every e_stride steps")
    parser.add_argument("--rel_tol", type=float, default=1e-3, help="Relative error used to measure convergence speed")
    parser.add_argument("--seed", type=int, default=31, help="Seed of the comparison (the model init uses the config seed)")
    parser.add_argument("--output", default=str(ROOT / "evaluate" / "output"), help="Parent folder of the results")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    if "type" not in cfg["sampler"]:
        raise ValueError("The config needs [sampler.metropolis] and [sampler.gibbs] subsections")

    print("JAX devices:", jax.devices())
    all_results = {}
    keys = jax.random.split(jax.random.PRNGKey(args.seed), len(args.N))
    for N, key in zip(args.N, keys):
        all_results[N] = run_N(cfg, N, args, key)

    run_dir = save_results(args, all_results)
    print(f"\nResults saved in {run_dir}")


if __name__ == "__main__":
    main()
