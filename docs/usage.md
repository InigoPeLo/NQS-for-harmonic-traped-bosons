# Usage guide

All commands run from the project root.

## 1. Train with the default config

```bash
uv run python train.py
```

A run goes through three phases:

1. **Thermalization:** `n_thermalization` sweeps from walkers drawn from `N(0, I)`. Two gates must pass:
   ```
   Thermalization: z = 0.84          # drift of <Σ|x|²> in σ; must be |z| < 3
   Thermalization: acceptance = 0.57 # must be in [0.4, 0.6]
   ```
2. **SR loop:** progress is printed every 10 iterations:
   ```
   Iteration 1/150: E = 4.909703, Var(E) = 1.991309, acceptance = 0.56
   Iteration 11/150: E = 4.124274, Var(E) = 0.222942, acceptance = 0.49
   ```
   `E` should decrease towards `E_0 = N` and `Var(E)` towards 0.
3. **Summary + save:**
   ```
   Final energy (mean of the last 15 iterations): E = 4.000001 +- 0.000011
   Exact energy: E_0 = 4.000000, relative error = 2.38e-07
   Final Var(E_loc) = 3.14e-05, alpha = 0.4865
   Results saved in results/N4_YYYYmmdd-HHMMSS-ffffff
   ```
   α ending slightly below 0.5 (the pure-Gaussian value `ω/2`) while E is exact is expected. The RBM can supply part of the quadratic decay, so only the *total* Gaussian width has to match `ω/2`. Use `Var(E_loc)` as the convergence check, not α.

## 2. Use your own config

Copy `config.toml`, edit it, and pass it with `--config`:

```bash
cp config.toml configs_n8.toml      # any name/location
# edit n_particles = 8, etc.
uv run python train.py --config configs_n8.toml
```

The file is copied into the run folder, so a run can always be reproduced.

### Example: 3D trap

Add `dim = 3` to `[system]` and leave everything else at the defaults:

```toml
[system]
n_particles = 4
dim = 3
```

For N = 4 this converges to the exact `E_0 = 4·3/2 = 6`:

```
Thermalization: acceptance = 0.47
Iteration 141/150: E = 5.999976, Var(E) = 0.000014, acceptance = 0.34
Final energy (mean of the last 15 iterations): E = 6.000006 +- 0.000010
Exact energy: E_0 = 6.000000, relative error = 1.03e-06
```

In 3D the acceptance passes the thermalization gate but falls to ~0.34 during training, because the wavefunction narrows and the gate is checked only once. A `step_size` of about 0.3 keeps it closer to 50%.

### Parameters

| Section.key | Default | Effect |
|---|---|---|
| `system.n_particles` | 4 | Number of bosons N. Exact energy `E_0 = N·dim·ω/2` (= N with the defaults) |
| `system.dim` | 2 *(not in file)* | Spatial dimension of each particle. Also sets the encoder input size. Put it **only** in `[system]` |
| `system.omega` | 1.0 *(not in file)* | Trap frequency. Can be added to `[system]` |
| `network.n_visible` | 32 | Latent size F of the Deep Sets encoder |
| `network.n_hidden_rbm` | 32 | Hidden units M of the amplitude RBM |
| `network.n_hidden_ffnn` | 32 | Hidden units K of the phase FFNN |
| `network.alpha` | 0.3 | Initial Gaussian width parameter (exact `ω/2`) |
| `network.init_scale` | 0.01 | Std of the initial RBM weights |
| `sampler.n_chains` | 256 | Parallel Markov chains |
| `sampler.step_size` | 0.4 | Metropolis step δ |
| `sampler.n_sweep` | 10 | Metropolis steps between recorded samples |
| `sr.learning_rate` | 0.05 | η |
| `sr.varepsilon` | 1e-3 | Diagonal shift ε on S |
| `training.n_samples` | 8 | Recorded samples per chain per iteration (`Ns = n_chains · n_samples`) |
| `training.n_thermalization` | 100 | Sweeps discarded before training |
| `training.n_iter` | 150 | SR iterations |
| `training.seed` | 0 | PRNG seed. Same seed and config give the same run on the same device |
| `training.output_dir` | `"results"` | Parent folder for run outputs |

Network size controls the cost: `p = (dim+1)F + M(F+1) + F + K(F+2) + 1` parameters, and SR scales as `O(p³)` per iteration. If you enlarge the network, also raise `n_chains · n_samples` towards `p`, or raise `varepsilon`.

## 3. Load a trained wavefunction

`train.py` saves θ flat. To rebuild the parameter pytree, use the run's own config:

```python
import tomllib, jax, jax.numpy as jnp
from src.nqs import NQS
from src.sr_optimizer import flatten_params

run = "results/N4_20261006-121907"
with open(f"{run}/config.toml", "rb") as f:
    cfg = tomllib.load(f)
data = jnp.load(f"{run}/results.npz")

dim = cfg["system"].get("dim", 2)   # same default as boson_trap
model = NQS(**cfg["network"], dim=dim)
_, unravel = flatten_params(model.init(jax.random.PRNGKey(0)))  # any key: only the structure is used
params = unravel(jnp.asarray(data["theta"]))

x = jnp.zeros((cfg["system"]["n_particles"], dim))
print(model.apply(params, x))                       # complex log ψ(x)
print(float(jax.nn.softplus(params["alpha_tilde"])))  # trained α
```

Run it with `uv run python your_script.py` from the root.

## 4. Analyse the training history

```python
import jax.numpy as jnp
d = jnp.load("results/N4_20261006-121907/results.npz")
d.files            # ['theta', 'energy', 'variance', 'acceptance', 'alpha']
d["energy"]        # (n_iter,) mean E_loc per iteration
d["variance"]      # (n_iter,) Var(E_loc) per iteration
d["acceptance"]    # (n_iter,) mean Metropolis acceptance per iteration
d["alpha"]         # (n_iter,) α after each update
```

For `.npz` files `jnp.load` defers to NumPy, so `d` is a NumPy `NpzFile` and each entry is a float32 NumPy array. `np.load` gives the same result, and runs saved before the switch to `jnp.savez` load identically. Wrap an entry in `jnp.asarray` to put it on the JAX device.

`variance` going to 0 is the cleanest convergence signal. For an exact eigenstate `E_loc` is constant, so `Var(E_loc) = 0` regardless of sampling noise.

## Edge cases and limits

- **`omega ≠ 1`.** The Hamiltonian and `exact_energy` handle it. The pure-Gaussian value of α becomes `ω/2`, so consider setting `network.alpha` near it.
- **Larger N or `dim`.** The all-particle move makes acceptance fall at a fixed `step_size`, so expect to lower `step_size` as `N·dim` grows. The Hessian-based Laplacian costs `(N·dim)²` per sample.
- **The final error bar is indicative only.** It treats the last 10% of iterations as independent.
- **Interrupting a run** (Ctrl-C) saves nothing, because results are written only at the end.

## Common errors

| Message | Meaning | Fix |
|---|---|---|
| `RuntimeError: Chains have not thermalized: z = … > 3.0` | `<Σ\|x\|²>` still drifting at the end of thermalization | Increase `training.n_thermalization` |
| `RuntimeError: Acceptance … too low (< 0.4)` | Steps too large, most moves rejected | Decrease `sampler.step_size` |
| `RuntimeError: Acceptance … too high (> 0.6)` | Steps too small, chains barely move | Increase `sampler.step_size` |
| `TypeError: … got an unexpected keyword argument '…'` | A config key does not match a dataclass field | Fix the spelling in the matching section (see [api.md](api.md)) |
| `TypeError: … got multiple values for keyword argument 'dim'` | `dim` was put in `[network]` | Move it to `[system]`. `train.py` passes it to `NQS` itself |
| `ValueError: Sum of sizes … must be equal to dimension 0 of the operand shape …` when loading a run | `NQS` rebuilt with a different `dim` (or network sizes) than the run | Pass `dim=cfg["system"].get("dim", 2)` and use the run's own `config.toml` |
| `KeyError: 'training'` / `'n_iter'` | Missing section or key in a custom config | All `[training]` keys are required. Start from `config.toml` |
| Energy oscillates or goes to NaN | SR step too aggressive, or S badly conditioned | Lower `sr.learning_rate`, raise `sr.varepsilon`, or increase `Ns` |
