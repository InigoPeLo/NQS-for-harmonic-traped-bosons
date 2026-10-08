"""Main script to train the NQS model to solve the boson trap problem using Stochastic Reconfiguration (SR) optimization."""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

from src.nqs import NQS
from src.sr_optimizer import SR, flatten_params
from src.boson_trap import boson_trap
from src.sampler_m import MetroSampler
from src.sampler_g import GibbsSampler
import argparse
import tomllib
import shutil
from datetime import datetime
from pathlib import Path



def load_config():
    """Read the path of the config file from the command line (default config.toml) and load it.
    Returns a nested dict, one entry per section: cfg["network"]["n_visible"], cfg["sr"]["learning_rate"],
    and the path of the file, so it can be copied next to the results."""

    parser = argparse.ArgumentParser(description="Train an NQS for N bosons in a 2D harmonic trap with SR")
    parser.add_argument("--config", default="config.toml", help="Path to the TOML file with the parameters")
    args = parser.parse_args()

    #tomllib needs the file opened in binary mode
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    return cfg, args.config


#Tag added at the end of the run folder name to know which sampler was used
SAMPLER_TAGS = {"metropolis": "m", "gibbs": "g"}

#Desired range of the final acceptance rate: outside it, train.py prints a warning at the end of the run
ACC_MIN, ACC_MAX = 0.25, 0.4


def save_results(output_dir, config_path, n_particles, sampler_type, theta, history):
    """Save a training run in its own folder output_dir/N{n_particles}_{date-time}_{m|g}/ with:
    results.npz: the final theta (flat parameters) and the history of every iteration
    config.toml: a copy of the config used, needed to rebuild the same NQS and to know how the run was done
    To rebuild the wavefunction: model = NQS(**cfg["network"], dim=system.dim), _, unravel = flatten_params(model.init(key))
    and params = unravel(data["theta"]). Returns the path of the folder."""

    #One folder per run, so we never overwrite previous results. The sampler tag goes at the end, so the
    #folders still sort by date: _m for Metropolis, _g for Gibbs
    run_dir = Path(output_dir) / (f"N{n_particles}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
                                  f"_{SAMPLER_TAGS[sampler_type]}")
    run_dir.mkdir(parents=True, exist_ok=True)

    #jnp.savez stores several named arrays in one file, read them back with jnp.load(path)["energy"]
    jnp.savez(run_dir / "results.npz", theta=jnp.asarray(theta),
             **{name: jnp.asarray(values) for name, values in history.items()})

    shutil.copy(config_path, run_dir / "config.toml")

    return run_dir


def make_train_step(log_psi, sample_fn, system, sr, n_samples):
    """Build the jitted SR iteration for a given model, sampler, system and optimizer.
    sample_fn(theta, walkers, n_samples, key) hides which sampler is used (see main).
    log_psi, sample_fn, system, sr and n_samples are taken from this enclosing function, so jax.jit
    treats them as constants and train_step only receives what changes every iteration."""

    @jax.jit
    def train_step(theta, walkers, key):
        """One SR iteration: sample with the current theta, compute E_loc and update theta.
       Returns the new theta, the new walkers, [energy, variance], the acceptance rate and the std of the phase."""

        #log_psi as a function of x only, with the theta of this iteration fixed
        log_psi_x = lambda x: log_psi(theta, x)

        #The walkers start from where the previous iteration left them, so they are almost thermalized
        samples, walkers, acceptance = sample_fn(theta, walkers, n_samples, key)

        #Local energy of every sample. It only evaluates psi, so it takes log_psi_x
        E_loc = system.batch_local_energy(log_psi_x, samples)

        #SR differentiates with respect to theta, so it takes log_psi(theta, x)
        theta_new, stats = sr.step(theta, log_psi, samples, E_loc)

        #Spread of the phase over the samples. It is 0 when the phase is constant, as it must be for the
        #bosonic ground state (a global phase does not matter, so we use the std and not the mean)
        phase_std = jnp.std(jnp.imag(jax.vmap(log_psi_x)(samples)))

        return theta_new, walkers, stats, acceptance, phase_std


    return train_step



def main():
    cfg, config_path = load_config()

    #The keys of each section match the fields of its class, so we can unpack them directly with **
    system = boson_trap(**cfg["system"])
    model = NQS(**cfg["network"], dim=system.dim)  #The dimension of the particles must match the system
    #The sampler is chosen in the config, and each one reads its own subsection
    samplers = {"metropolis": MetroSampler, "gibbs": GibbsSampler}
    sampler_type = cfg["sampler"]["type"]
    sampler = samplers[sampler_type](**cfg["sampler"][sampler_type])
    sr = SR(**cfg["sr"])

    #The training parameters are not fields of any class, we read them one by one
    train_cfg = cfg["training"]

    key = jax.random.PRNGKey(train_cfg["seed"])
    key_params, key_walkers, key_train, key_therm = jax.random.split(key, 4)
    #Initialize the model parameters with the key_params
    params = model.init(key_params)

    #Initialize the walkers with the key_walkers
    walkers = sampler.init_walkers(key_walkers, system.n_particles, system.dim)

    #We flatten the parameters of the model to get theta, and we also get the unravel function to go back to the original structure
    theta, unravel = flatten_params(params)

    #We build the log_psi function that takes theta and x as arguments, and returns log(psi(theta, x))
    log_psi = lambda t, x: model.apply(unravel(t), x)    

    #Both samplers sample |psi|^2, but Metropolis needs log_psi(x) and Gibbs needs the parameters dictionary.
    #sample_fn gives both the same signature, so the rest of the script does not depend on the sampler
    if sampler_type == "gibbs":
        sample_fn = lambda t, w, n, k: sampler.sample(unravel(t), w, n, k)
    else:
        sample_fn = lambda t, w, n, k: sampler.sample(lambda x: log_psi(t, x), w, n, k)

    #We need to thermalize the walkers before starting the training,
    #we do n_thermalization sweeps and discard the results
    sample_therm, walkers, accepts_mean = sample_fn(theta, walkers, train_cfg["n_thermalization"], key_therm)

    #We check if the chains have thermalized, and raise an error if not.
    thermalized, z=sampler.check_therm(sample_therm)
    print(f"Thermalization: z = {float(z):.2f}")
    if not thermalized:
        raise RuntimeError(f"Chains have not thermalized: z = {float(z):.2f} > 3.0, increase n_thermalization in the config")



    #Now we can make the trainig loop

    train_step = make_train_step(log_psi, sample_fn, system, sr, train_cfg["n_samples"])

    #Empty lists of ebergy, variance and acceptance to store the results of every iteration
    energy_list = []
    variance_list = []
    acceptance_list = []
    alpha_list = []
    phase_std_list=[]

    for n in range(train_cfg["n_iter"]):
        key_train, key_iter = jax.random.split(key_train)
        theta, walkers, (energy, variance), acceptance, phase_std= train_step(theta, walkers, key_iter)
        energy_list.append(energy)
        variance_list.append(variance)
        acceptance_list.append(acceptance)
        #alpha of the Gaussian envelope after this update
        alpha_list.append(jax.nn.softplus(unravel(theta)["alpha_tilde"]))
        phase_std_list.append(phase_std)

        if n % 10 == 0 or n == train_cfg["n_iter"] - 1:
            print(f"Iteration {n+1}/{train_cfg['n_iter']}: E = {float(energy):.6f}, Var(E) = {float(variance):.6f}, acceptance = {float(acceptance):.2f}")

    print(f"Training finished\n")

    #Every energy of the list is noisy (only Ns samples), so we average the last 10% of the iterations.
    #The error bar ignores the correlation between iterations, so it is only indicative: the proper
    #value comes from a long run at fixed theta (final evaluation)
    n_last = max(1, train_cfg["n_iter"] // 10)
    last_energies = jnp.array(energy_list[-n_last:])
    E_final = float(jnp.mean(last_energies))
    E_error = float(jnp.std(last_energies) / jnp.sqrt(n_last))
    
    print(f"Final energy (mean of the last {n_last} iterations): E = {E_final:.6f} +- {E_error:.6f}")
    print(f"Exact energy: E_0 = {system.exact_energy:.6f}, relative error = {abs(E_final - system.exact_energy) / system.exact_energy:.2e}")
    print(f"Final Var(E_loc) = {float(variance_list[-1]):.2e}, alpha = {float(alpha_list[-1]):.4f}")

    #The acceptance drops while psi narrows and settles once it has converged, so we check the mean of the
    #last iterations. A low acceptance does not bias E, it makes the samples more correlated, so it is only a warning
    n_acc = min(20, len(acceptance_list))
    acc_final = float(jnp.mean(jnp.array(acceptance_list[-n_acc:])))
    print(f"Final acceptance (mean of the last {n_acc} iterations) = {acc_final:.2f}")
    if acc_final < ACC_MIN:
        print(f"Warning: final acceptance {acc_final:.2f} below {ACC_MIN}, decrease step_size in the config for the next run")
    elif acc_final > ACC_MAX:
        print(f"Warning: final acceptance {acc_final:.2f} above {ACC_MAX}, increase step_size in the config for the next run")

    #Save theta, the history and a copy of the config to analyse the run later without training again
    history = {"energy": energy_list, "variance": variance_list, "acceptance": acceptance_list,
               "alpha": alpha_list, "phase_std": phase_std_list}
    run_dir = save_results(train_cfg["output_dir"], config_path, system.n_particles, sampler_type, theta, history)
    print(f"Results saved in {run_dir}")

    #The trained wavefunction is the model (architecture) together with its parameters
    return model, unravel(theta)


if __name__ == "__main__":
    main()