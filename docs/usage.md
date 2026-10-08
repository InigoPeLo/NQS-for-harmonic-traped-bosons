# Usage guide

All commands run from the project root.

## 1. Train with the default config

```bash
uv run python train.py
```

The shipped `config.toml` is set to N = 20 (see [Parameters](#parameters)). The output below comes from the N = 4 reference settings, where the exact energy is `E_0 = 4`.

A run goes through three phases:

1. **Thermalization:** `n_thermalization` sweeps from walkers drawn from `N(0, I)`. The run aborts unless the chains have stopped drifting:
   ```
   Thermalization: z = 0.84          # drift of <Σ|x|²> in σ; must be |z| < 3
   ```
   The acceptance is not checked here. ψ is still the initial one, and the acceptance only drops from this point (see step 3).
2. **SR loop:** progress is printed every 10 iterations and at the last one:
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
   Final acceptance (mean of the last 20 iterations) = 0.45
   Results saved in results/N4_YYYYmmdd-HHMMSS-ffffff_m
   ```
   If the final acceptance is outside `[0.25, 0.4]`, an extra line says which way to change `step_size` **for the next run**, for example `Warning: final acceptance 0.15 below 0.25, decrease step_size in the config for the next run`. The run is still saved: a low acceptance makes the samples more correlated but does not bias E.

   The acceptance falls during training because ψ narrows as α grows, and it settles once ψ has converged: 0.56 → 0.45 at N = 4. Choose `step_size` for the *final* value, not the initial one.

   α ending slightly below 0.5 (the pure-Gaussian value `ω/2`) while E is exact is expected. The RBM can supply part of the quadratic decay, so only the *total* Gaussian width has to match `ω/2`. Use `Var(E_loc)` as the convergence check, not α.

## 2. Use your own config

Copy `config.toml`, edit it, and pass it with `--config`:

```bash
cp config.toml configs_n8.toml      # any name/location
# edit n_particles = 8, etc.
uv run python train.py --config configs_n8.toml
```

The file is copied into the run folder, so a run can always be reproduced.

### Choose the sampler

Set `type` in `[sampler]`. Each sampler keeps its own settings, so switching is a one-line change:

```toml
[sampler]
type = "gibbs"          # or "metropolis"

[sampler.metropolis]    # used when type = "metropolis"
n_chains = 256
step_size = 0.14
n_sweep = 10

[sampler.gibbs]         # used when type = "gibbs"
n_chains = 256
step_size = 0.5
n_sweep = 10
n_metro = 3
```

- **`metropolis`** moves all particles at once. Lower `step_size` as `N·dim` grows.
- **`gibbs`** samples the RBM hidden units, then moves every particle on its own (`n_metro` moves per Gibbs step). `step_size` is a single-particle step and does not need to shrink with N. Its acceptance is per particle and runs higher than the Metropolis one at the same settings.

Both converge to the same energy. At N = 20, 100 iterations, with the shipped settings:

```
metropolis: Final energy (mean of the last 10 iterations): E = 20.096210 +- 0.003635   acceptance 0.54
gibbs:      Final energy (mean of the last 10 iterations): E = 20.089064 +- 0.003092   acceptance 0.67
```

With these settings both final acceptances (0.54 Metropolis, 0.67 Gibbs) trigger the `above 0.4` warning: raise `step_size` to bring them into `[0.25, 0.4]`. The band is the same for both samplers, so with Gibbs raise `step_size` if you want to stay inside it.

### Example: 3D trap

Add `dim = 3` to `[system]` and leave everything else at the defaults:

```toml
[system]
n_particles = 4
dim = 3
```

For N = 4 this converges to the exact `E_0 = 4·3/2 = 6`:

```
Iteration 1/150: E = 7.396373, Var(E) = 3.191899, acceptance = 0.47
Iteration 141/150: E = 5.999976, Var(E) = 0.000014, acceptance = 0.34
Final energy (mean of the last 15 iterations): E = 6.000006 +- 0.000010
Exact energy: E_0 = 6.000000, relative error = 1.03e-06
```

In 3D the acceptance falls from 0.47 to ~0.34 during training, so the final acceptance warning fires with `step_size = 0.4`, even though the energy is exact. A `step_size` of about 0.3 keeps it closer to 50%.

### Example: N = 40

The N = 4 SR settings diverge at N = 40: the energy drops *below* `E_0` within 2–3 iterations, then turns NaN. The scale of S grows ~N² (see [decisions.md](decisions.md#sr-solver-dense-solve-with-diagonal-shift)), so `varepsilon` must grow and `learning_rate` shrink. Settings tested at N = 40, 2D, 150 iterations:

| `varepsilon` | `learning_rate` | Final E (`E_0 = 40`) | Final `Var(E)` |
|---|---|---|---|
| 1e-3 | 0.05, 0.01, 0.005 | NaN | NaN |
| 1 | 0.001 | 62.9 | 63 |
| 1 | 0.01 | 40.71 | 1.04 |
| **1** | **0.02** | **40.07** | **0.16** |

```toml
[system]
n_particles = 40

[sampler.metropolis]
step_size = 0.15

[sr]
learning_rate = 0.02
varepsilon = 1.0

[training]
n_thermalization = 500
n_iter = 300          # at 150 iterations it is still improving (relative error 1.8e-3)
```

With `step_size = 0.15`, the acceptance ends at ~0.34 and the final warning fires. Keep the N = 4 settings and the N = 40 settings in separate files (e.g. `configs/n40.toml`), since each needs its own SR values. Each run takes ~30 s per 150 iterations on an RTX 5070 Laptop GPU.

### Parameters

The first column holds the N = 4 reference settings used in the examples of this guide. The second holds the values in the shipped `config.toml`. For N = 40, see the example above.

| Section.key | N = 4 value | `config.toml` | Effect |
|---|---|---|---|
| `system.n_particles` | 4 | 20 | Number of bosons N. Exact energy `E_0 = N·dim·ω/2` (= N with the defaults) |
| `system.dim` | 2 *(not in file)* | 2 *(not in file)* | Spatial dimension of each particle. Also sets the encoder input size. Put it **only** in `[system]` |
| `system.omega` | 1.0 *(not in file)* | 1.0 *(not in file)* | Trap frequency. Can be added to `[system]` |
| `network.n_visible` | 32 | 32 | Latent size F of the Deep Sets encoder |
| `network.n_hidden_rbm` | 32 | 32 | Hidden units M of the RBM that models \|ψ\|² |
| `network.n_hidden_ffnn` | 32 | 32 | Hidden units K of the phase FFNN |
| `network.alpha` | 0.3 | 0.3 | Initial Gaussian width parameter (exact `ω/2`) |
| `network.init_scale` | 0.01 | 0.01 | Std of the initial RBM weights |
| `sampler.type` | `"metropolis"` | `"metropolis"` | Which sampler to use: `"metropolis"` or `"gibbs"` |
| `sampler.metropolis.n_chains` | 256 | 256 | Parallel Markov chains |
| `sampler.metropolis.step_size` | 0.4 | 0.14 | Metropolis step δ (all particles at once). Choose it so the *final* acceptance is in `[0.25, 0.4]` |
| `sampler.metropolis.n_sweep` | 10 | 10 | Metropolis steps between recorded samples |
| `sampler.gibbs.n_chains` | — | 256 | Parallel Markov chains |
| `sampler.gibbs.step_size` | — | 0.5 | Single-particle step of the Metropolis moves inside Gibbs |
| `sampler.gibbs.n_sweep` | — | 10 | Gibbs steps between recorded samples |
| `sampler.gibbs.n_metro` | — | 3 | Metropolis moves of every particle per Gibbs step (class default 1) |
| `sr.learning_rate` | 0.05 | 0.02 | η |
| `sr.varepsilon` | 1e-3 | 0.1 | Diagonal shift ε on S. Absolute, so it must grow with N (1.0 at N = 40). Write it as a float (`1.0`) |
| `training.n_samples` | 8 | 10 | Recorded samples per chain per iteration (`Ns = n_chains · n_samples`) |
| `training.n_thermalization` | 100 | 200 | Sweeps discarded before training |
| `training.n_iter` | 150 | 400 | SR iterations |
| `training.seed` | 0 | 1 | PRNG seed. Same seed and config give the same run on the same device |
| `training.output_dir` | `"results"` | `"results"` | Parent folder for run outputs |

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
d.files            # ['theta', 'energy', 'variance', 'acceptance', 'alpha', 'phase_std']
d["energy"]        # (n_iter,) mean E_loc per iteration
d["variance"]      # (n_iter,) Var(E_loc) per iteration
d["acceptance"]    # (n_iter,) mean Metropolis acceptance per iteration
d["alpha"]         # (n_iter,) α after each update
d["phase_std"]     # (n_iter,) std of the phase Im log ψ over the samples of each iteration
```

Older runs have no `phase_std`. Check `"phase_std" in d.files` before reading it.

`phase_std` should go to 0, because the bosonic ground state has a constant phase. A phase that varies over the samples costs `½<|∇φ|²>` of energy and adds to the imaginary part of `E_loc`.

For `.npz` files `jnp.load` defers to NumPy, so `d` is a NumPy `NpzFile` and each entry is a float32 NumPy array. `np.load` gives the same result, and runs saved before the switch to `jnp.savez` load identically. Wrap an entry in `jnp.asarray` to put it on the JAX device.

`variance` going to 0 is the cleanest convergence signal. For an exact eigenstate `E_loc` is constant, so `Var(E_loc) = 0` regardless of sampling noise.

`variance` is the variance of the **complex** `E_loc`, so it includes the imaginary part. The imaginary part averages to 0 but is not constant unless the learned phase is exactly flat. In the default 2D run, the fixed-θ evaluation of the notebook splits the final value as `3.0e-06` (real) + `3.3e-05` (imaginary). In other words, the floor seen at the end of training comes mostly from the phase network, not from errors in the amplitude.

## 5. Analysis notebook

`notebooks/analysis.ipynb` loads every run in the folders of `RESULTS_DIRS` (default `[ROOT / "results"]`; add more to compare runs saved elsewhere). Open it with the `.venv` kernel; re-running it picks up new runs. Its sections:

| Section | What it shows | What to look for |
|---|---|---|
| 1. Load the runs | Numbered list of all runs, sorted by date, with N, d, sampler (`m`/`g`), δ, η, ε, iterations, `E_final` and status (`converged`, `rel err …`, `DIVERGED` for NaN/inf) | Which runs exist and which `#` to select |
| 1b. Choose the runs | The filters below. Produces `runs`, used by every later section | — |
| Summary table | `E_final ± err`, relative error, `Var(E)`, final acceptance, α, iterations to reach `REL_TOL = 1e-3` | Same numbers as the `train.py` summary, for the selected runs side by side |
| 2. Training curves | `E/E_0`, relative error, `Var(E_loc)`, acceptance (green band `[0.25, 0.4]`), α, relative error vs variance | `Var(E_loc)` falling; acceptance staying in the band |
| 3. Acceptance vs convergence | Acceptance drift per run against iterations to `REL_TOL` and final variance | Whether low final acceptance correlates with slower convergence |
| 4. Trained wavefunction | Cut of `log\|ψ\|` and phase with particle 0 on the x axis, vs exact `−ω x²/2` | Default run: max deviation `2.9e-04`, phase flat to `1.2e-03` for \|x\| < 2 |
| 5. Final evaluation at fixed θ | Fresh sampling (`N_THERM = N_EVAL = 200` sweeps), `E ± err` from per-chain means, `Var` split into real and imaginary parts, one-body density | Default run: `E = 3.999992 ± 0.000008` (−1.0σ from `E_0`), `<\|x_i\|²> = 0.998` vs exact `1.000` |
| 6. Is the phase constant? | For the run of section 5: histogram of `φ − <φ>`. For every selected run: `std(φ)`, its energy cost `½<\|∇φ\|²>` and its share of `E − E_0` (re-sampled with `N_THERM_PHASE = 200`, `N_PHASE = 20` sweeps). Then `phase_std` against SR iteration | A single spike at 0 and a cost well below `E − E_0`. Runs without `phase_std` are listed and skipped in the last plot |

- **Selecting runs (section 1b):** filters left as `None` are ignored, and the ones that are set must all pass.

  ```python
  SELECT_INDEX  = [0, 4]                                  # by # in the list of section 1
  SELECT_NAMES  = ["N4_20261006-121907"]                  # by folder name (unknown names are reported)
  SELECT_N      = [40]                                    # by number of particles
  SELECT_DIM    = [2]                                     # by dimension
  SELECT_SAMPLER = ["gibbs"]                              # by sampler: "metropolis" and/or "gibbs"
  SELECT_WHERE  = lambda r: r["cfg"]["sr"]["learning_rate"] >= 0.01   # any condition on a run
  SELECT_LAST   = 3                                       # most recent ones, after the other filters
  SKIP_DIVERGED = True                                    # drop runs with NaN/inf (default)
  ```

  If no run passes, the cell raises `ValueError: No run passes the filters`.
- **Choosing the run for sections 4–5:** `RUN_NAME = "N4_20261006-121907"`, or `None` for the most recent selected run. It must be among the selected runs and not diverged; otherwise the cell raises a `ValueError` saying why.
- **Memory:** section 5 evaluates `E_loc` in batches of `n_chains · n_samples` configurations, the size of one training iteration. All 51 200 samples at once do not fit in 8 GB of GPU memory at N = 40, because each needs an `(N·dim)²` Hessian.
- **Error bar:** section 5 is the proper energy estimate. Chains are independent, so the spread of chain means gives an error bar that accounts for autocorrelation within each chain. The `train.py` figure does not.
- **Precision:** raise `N_EVAL` for a smaller error bar. The error scales as `1/√N_EVAL`, and the cost grows linearly with it.
- **Both samplers:** `sampler_config` reads the run's sampler from its `config.toml`. A flat `[sampler]` section (runs saved before the Gibbs sampler) means Metropolis. The run list, summary table and plot labels show it as `m` or `g`, and sections 5–6 re-sample with the same sampler and settings the run was trained with (`build_sampler`). The acceptance panel is per particle for Gibbs runs, so compare it only between runs of the same sampler.
- **Runs from before the |ψ|² RBM:** sections 1–3 only read the saved history, so they work for any run. Sections 4–6 rebuild ψ with the current `NQS`, which halves the RBM output. A run trained when the RBM modelled |ψ| would therefore be analysed with the wrong wavefunction. Analyse those runs with the code version that produced them.

## Edge cases and limits

- **`omega ≠ 1`.** The Hamiltonian and `exact_energy` handle it. The pure-Gaussian value of α becomes `ω/2`, so consider setting `network.alpha` near it.
- **Larger N or `dim`.** With Metropolis, the all-particle move makes acceptance fall at a fixed `step_size`, so expect to lower `step_size` as `N·dim` grows; Gibbs moves one particle at a time and avoids this. With either sampler, the Hessian-based Laplacian costs `(N·dim)²` per sample.
- **The final error bar is indicative only.** It treats the last 10% of iterations as independent.
- **Interrupting a run** (Ctrl-C) saves nothing, because results are written only at the end.

## Common errors

| Message | Meaning | Fix |
|---|---|---|
| `RuntimeError: Chains have not thermalized: z = … > 3.0` | `<Σ\|x\|²>` still drifting at the end of thermalization (the test is on `\|z\|`, so it also fires for negative `z`) | Increase `training.n_thermalization` |
| `ValueError: check_therm needs at least 4 samples per chain, got …` | `training.n_thermalization < 4`, too few sweeps to compare the third and fourth quarters | Set `n_thermalization ≥ 4` (default 100) |
| `Warning: final acceptance … below 0.25` (run still saved) | Steps too large for the converged ψ: samples are strongly correlated | Decrease `sampler.step_size` for the next run |
| `Warning: final acceptance … above 0.4` (run still saved) | Steps too small: chains barely move | Increase `sampler.step_size` for the next run |
| `TypeError: … got an unexpected keyword argument '…'` | A config key does not match a dataclass field | Fix the spelling in the matching section (see [api.md](api.md)) |
| `TypeError: … got multiple values for keyword argument 'dim'` | `dim` was put in `[network]` | Move it to `[system]`. `train.py` passes it to `NQS` itself |
| `ValueError: Sum of sizes … must be equal to dimension 0 of the operand shape …` when loading a run | `NQS` rebuilt with a different `dim` (or network sizes) than the run | Pass `dim=cfg["system"].get("dim", 2)` and use the run's own `config.toml` |
| `KeyError: 'training'` / `'n_iter'` | Missing section or key in a custom config | All `[training]` keys are required. Start from `config.toml` |
| `KeyError: 'type'` | Config with the old flat `[sampler]` section (e.g. the `config.toml` copied into a run saved before the Gibbs sampler) | Add `type = "metropolis"` to `[sampler]` and move its keys to `[sampler.metropolis]` |
| `KeyError: 'gibbs'` / `'metropolis'` | `type` names a subsection that is missing | Add the `[sampler.<type>]` subsection |
| `TypeError: … unexpected keyword argument 'n_metro'` | `n_metro` put in `[sampler.metropolis]` | It only exists in `[sampler.gibbs]` |
| Energy oscillates or goes to NaN | SR step too aggressive, or S badly conditioned | Lower `sr.learning_rate`, raise `sr.varepsilon`, or increase `Ns` |
| Energy goes *below* `E_0`, then NaN, at large N | `varepsilon` too small for the scale of S, which grows ~N² | Raise `sr.varepsilon` (1.0 at N = 40) and lower `learning_rate` (0.02 at N = 40) |
