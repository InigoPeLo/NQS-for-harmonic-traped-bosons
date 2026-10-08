# API

The project has no HTTP interface. Its public surface is the `train.py` CLI and the five `src` modules.

## CLI

```
uv run python train.py [--config PATH]
```

| Option | Default | Description |
|---|---|---|
| `--config` | `config.toml` | TOML file with `[system]`, `[network]`, `[sampler]` (`type` + `[sampler.metropolis]` / `[sampler.gibbs]`), `[sr]`, `[training]` |

**Output:** creates `{training.output_dir}/N{n_particles}_{YYYYmmdd-HHMMSS-ffffff}_{m|g}/` (`_m` Metropolis, `_g` Gibbs) containing `results.npz` and `config.toml`. `main()` also returns `(model, params)` when imported.

**Exit with error (`RuntimeError`)** if the thermalization drift is `|z| ≥ 3`. If the mean acceptance of the last 20 iterations is outside `[0.25, 0.4]` (`ACC_MIN`, `ACC_MAX` in `train.py`), it prints a `Warning:` line but still saves the run. See [usage.md](usage.md#common-errors).

## Conventions

- **Configuration** `X`: array `(N, dim)`. **Batch**: `(B, N, dim)`.
- **`log_psi`** is always a callable returning a complex scalar `log ψ = log|ψ| + iφ`:
  - `log_psi(x)`: θ fixed. Used by `boson_trap` and `MetroSampler`.
  - `log_psi(theta, x)`: θ is a flat real vector. Used by `SR`.
- **`GibbsSampler`** takes no `log_psi`: it takes the `NQS` parameter dict (`unravel(theta)`).
- **Keys** are `jax.random.PRNGKey` values, and every random method takes one explicitly.
- All classes are frozen dataclasses. Their fields are exactly the config keys of the matching section.

---

## `src.boson_trap`

### `boson_trap(n_particles, dim=2, omega=1.0)`

| Member | Signature | Returns |
|---|---|---|
| `exact_energy` | property | `n_particles · dim · omega / 2` |
| `trap_potential` | `(x)` | `½ ω² Σ x²` (real scalar) |
| `kinetic_local` | `(log_psi, x)` | `−½(∇²f + ∇f·∇f)`, complex scalar |
| `local_energy` | `(log_psi, x)` | `kinetic_local + trap_potential` |
| `batch_local_energy` | `(log_psi, x)` with `x: (B, N, dim)` | `(B,)` complex |

```python
system = boson_trap(n_particles=4)
system.exact_energy                        # 4.0
E_loc = system.batch_local_energy(lambda x: model.apply(params, x), samples)
```

---

## `src.nqs`

All four classes have `init(key) -> params` (a dict pytree) and `apply(params, input)`.

| Class | Fields | `apply` input → output |
|---|---|---|
| `DSE` | `n_neurons`, `dim=2` | `x (N, dim)` → `H (n_neurons,)`, `Σ_i swish(W x_i + b)` |
| `RBM` | `n_visible`, `n_hidden`, `init_scale` | `H` → real scalar, `a·H + Σ log cosh(b + W H)`, the log of \|ψ\|² without the envelope |
| `FFNN` | `n_visible`, `n_hidden` | `H` → real scalar, `u · log cosh(V H + c)` |
| `NQS` | `n_visible`, `n_hidden_rbm`, `n_hidden_ffnn`, `alpha`, `init_scale=0.01`, `dim=2` | `x (N, dim)` → complex `log ψ = ½·RBM(H) − α Σ\|x_i\|² + i·FFNN(H)` |

`NQS.dim` must equal `boson_trap.dim`. `train.py` enforces this by building the model as `NQS(**cfg["network"], dim=system.dim)`, so `dim` must **not** appear in `[network]`. If it does, Python raises `TypeError: got multiple values for keyword argument 'dim'`.

Structure of `NQS` params:

```python
{
  "params_dse":  {"W": (F, dim), "b": (F,)},
  "params_rbm":  {"W": (M, F), "b": (M,), "a": (F,)},
  "params_ffnn": {"V": (K, F), "u": (K,), "c": (K,)},
  "alpha_tilde": ()            # α = softplus(alpha_tilde)
}
```

```python
model = NQS(n_visible=32, n_hidden_rbm=32, n_hidden_ffnn=32, alpha=0.3)
params = model.init(jax.random.PRNGKey(0))
model.apply(params, jnp.zeros((4, 2)))     # complex scalar
```

---

## `src.sampler_m`

### `MetroSampler(n_chains, step_size=0.4, n_sweep=10)`

| Method | Signature | Returns |
|---|---|---|
| `init_walkers` | `(key, n_particles, dim)` | `(n_chains, N, dim)` drawn from `N(0, I)` |
| `step` | `(log_psi, (walkers, log_prob), key)` | `((walkers, log_prob), accept (n_chains,))`, one Metropolis step in `lax.scan` form |
| `sample` | `(log_psi, walkers, n_samples, key)` | `(samples (n_samples·n_chains, N, dim), walkers, mean_acceptance)` |
| `check_therm` | `(samples, z_max=3.0)` | `(thermalized: bool, z)`. Raises `ValueError` with fewer than 4 samples per chain |

`sample` runs `n_samples` sweeps of `n_sweep` steps and records one configuration per chain per sweep. Sample order is sweep-major (`reshape` of `(n_samples, n_chains, …)`), and `check_therm` relies on that order.

```python
sampler = MetroSampler(n_chains=256)
walkers = sampler.init_walkers(key, 4, 2)
samples, walkers, acc = sampler.sample(lambda x: model.apply(params, x), walkers, 8, key2)
```

---

## `src.sampler_g`

### `GibbsSampler(n_chains, step_size=0.4, n_sweep=10, n_metro=1)`

Block Gibbs sampler of |ψ|² for the `NQS` of `src.nqs`. `params` is always the `NQS` parameter dict.

| Method | Signature | Returns |
|---|---|---|
| `init_walkers` | `(key, n_particles, dim)` | `(n_chains, N, dim)` drawn from `N(0, I)` |
| `hidden_sample` | `(params, walkers, key)` | `h (n_chains, M)` in `{−1, +1}`, with `p(h_j = +1 \| x) = sigmoid(2θ_j)` |
| `log_p1` | `(params, c, walkers)` with `c (n_chains, F)` | `(n_chains, N)`, unnormalized `c·swish(W_d y + b_d) − 2α\|y\|²` per particle |
| `metro_step` | `(params, c, (walkers, log_p), key)` | `((walkers, log_p), accept (n_chains, N))`, one per-particle Metropolis step with `c` fixed |
| `step` | `(params, walkers, key)` | `(walkers, mean_acceptance)`, one Gibbs step: `h \| x`, `c = a + h·W`, then `n_metro` × `metro_step` |
| `sample` | `(params, walkers, n_samples, key)` | `(samples (n_samples·n_chains, N, dim), walkers, mean_acceptance)`, same layout as `MetroSampler.sample` |
| `check_therm` | `(samples, z_max=3.0)` | Same as `MetroSampler.check_therm` |

`n_sweep` Gibbs steps are made between recorded samples. The acceptance is per particle and per Metropolis step.

```python
sampler = GibbsSampler(n_chains=256, step_size=0.5, n_metro=3)
walkers = sampler.init_walkers(key, 4, 2)
samples, walkers, acc = sampler.sample(params, walkers, 8, key2)
```

---

## `src.sr_optimizer`

| Function / class | Signature | Returns |
|---|---|---|
| `flatten_params` | `(params)` | `(theta (p,), unravel)` via `ravel_pytree` |
| `log_derivatives` | `(theta, log_psi, samples)` | `O (Ns, p)` complex, `O_k = ∂_θk log ψ` |
| `compute_S_F` | `(O, E_loc)` | `S (p, p)`, `F (p,)`, both complex covariances, sliced from one joint covariance of `(O, E_loc)` |
| `SR(learning_rate, varepsilon=1e-4)` | dataclass | — |
| `SR.step` | `(theta, log_psi, samples, E_loc)` | `(theta_new, [mean(Re E_loc), Var(E_loc)])`, where `Var` is over the complex `E_loc` (real + imaginary parts) |

`SR.step` solves `(Re S + εI) δθ = −η Re F` and returns `θ + δθ`.

```python
theta, unravel = flatten_params(params)
log_psi = lambda t, x: model.apply(unravel(t), x)
sr = SR(learning_rate=0.05, varepsilon=1e-3)
theta, (E, var) = sr.step(theta, log_psi, samples, E_loc)
```
