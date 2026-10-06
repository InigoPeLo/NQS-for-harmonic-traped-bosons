"""Main script to train the NQS model to solve the boson trap problem using Stochastic Reconfiguration (SR) optimization."""

import jax
import jax.numpy as jnp

from src.nqs import NQS
from src.sr_optimizer import SR, flatten_params
from src.boson_trap import boson_trap
from src.sampler import MetroSampler
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


def save_results(output_dir, config_path, n_particles, theta, history):
    """Save a training run in its own folder output_dir/N{n_particles}_{date-time}/ with:
    results.npz: the final theta (flat parameters) and the history of every iteration
    config.toml: a copy of the config used, needed to rebuild the same NQS and to know how the run was done
    To rebuild the wavefunction: model = NQS(**cfg["network"]), _, unravel = flatten_params(model.init(key))
    and params = unravel(data["theta"]). Returns the path of the folder."""

    #One folder per run, so we never overwrite previous results
    run_dir = Path(output_dir) / f"N{n_particles}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    #jnp.savez stores several named arrays in one file, read them back with jnp.load(path)["energy"]
    jnp.savez(run_dir / "results.npz", theta=jnp.asarray(theta),
             **{name: jnp.asarray(values) for name, values in history.items()})

    shutil.copy(config_path, run_dir / "config.toml")

    return run_dir


def make_train_step(log_psi, sampler, system, sr, n_samples):
    """Build the jitted SR iteration for a given model, sampler, system and optimizer.
    log_psi, sampler, system, sr and n_samples are taken from this enclosing function, so jax.jit
    treats them as constants and train_step only receives what changes every iteration."""

    @jax.jit
    def train_step(theta, walkers, key):
        """One SR iteration: sample with the current theta, compute E_loc and update theta.
        Returns the new theta, the new walkers, [energy, variance] and the acceptance rate."""

        #log_psi as a function of x only, with the theta of this iteration fixed
        log_psi_x = lambda x: log_psi(theta, x)

        #The walkers start from where the previous iteration left them, so they are almost thermalized
        samples, walkers, acceptance = sampler.sample(log_psi_x, walkers, n_samples, key)

        #Local energy of every sample. It only evaluates psi, so it takes log_psi_x
        E_loc = system.batch_local_energy(log_psi_x, samples)

        #SR differentiates with respect to theta, so it takes log_psi(theta, x)
        theta_new, stats = sr.step(theta, log_psi, samples, E_loc)

        return theta_new, walkers, stats, acceptance

    return train_step



def main():
    cfg, config_path = load_config()

    #The keys of each section match the fields of its class, so we can unpack them directly with **
    system = boson_trap(**cfg["system"])
    model = NQS(**cfg["network"])
    sampler = MetroSampler(**cfg["sampler"])
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

    #We need to termalize the walkers before starting the training,
    #we do n_termalization sweeps and discard the results

    #We evaluate log_psi for the initial walkers to get the starting log_prob
    log_psi_x = lambda x: log_psi(theta, x)

    sample_therm, walkers, accepts_mean=sampler.sample(log_psi_x, walkers, train_cfg["n_thermalization"], key_therm)

    #We check if the chains have thermalized, and raise an error if not.
    thermalized, z=sampler.check_therm(sample_therm)
    print(f"Thermalization: z = {float(z):.2f}")
    if not thermalized:
        raise RuntimeError(f"Chains have not thermalized: z = {float(z):.2f} > 3.0, rise n_thermalization in the config")

    #The acceptance rate must be close to 50%: if it is too low the step is too large and almost every move
    #is rejected, if it is too high the step is too small and the chains barely move. Both give long autocorrelations
    print(f"Thermalization: acceptance = {float(accepts_mean):.2f}")
    if accepts_mean < 0.4:
        raise RuntimeError(f"Acceptance {float(accepts_mean):.2f} too low (< 0.4): decrease step_size in the config")
    if accepts_mean > 0.6:
        raise RuntimeError(f"Acceptance {float(accepts_mean):.2f} too high (> 0.6): increase step_size in the config")

    #Now we can make the trainig loop

    train_step = make_train_step(log_psi, sampler, system, sr, train_cfg["n_samples"])

    #Empty lists of ebergy, variance and acceptance to store the results of every iteration
    energy_list = []
    variance_list = []
    acceptance_list = []
    alpha_list = []

    for n in range(train_cfg["n_iter"]):
        key_train, key_iter = jax.random.split(key_train)
        theta, walkers, (energy, variance), acceptance = train_step(theta, walkers, key_iter)
        energy_list.append(energy)
        variance_list.append(variance)
        acceptance_list.append(acceptance)
        #alpha of the Gaussian envelope after this update, it should tend to 0.5
        alpha_list.append(jax.nn.softplus(unravel(theta)["alpha_tilde"]))

        if n % 10==0:
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
    print(f"Final Var(E_loc) = {float(variance_list[-1]):.2e}, alpha = {float(alpha_list[-1]):.4f} (exact 0.5)")

    #Save theta, the history and a copy of the config to analyse the run later without training again
    history = {"energy": energy_list, "variance": variance_list, "acceptance": acceptance_list, "alpha": alpha_list}
    run_dir = save_results(train_cfg["output_dir"], config_path, system.n_particles, theta, history)
    print(f"Results saved in {run_dir}")

    #The trained wavefunction is the model (architecture) together with its parameters
    return model, unravel(theta)




if __name__ == "__main__":
    main()