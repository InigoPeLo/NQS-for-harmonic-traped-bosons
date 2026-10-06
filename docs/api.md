# API

The project has no HTTP interface. Its public surface is the `train.py` CLI and the four `src` modules.

## CLI

```
uv run python train.py [--config PATH]
```

| Option | Default | Description |
|---|---|---|
| `--config` | `config.toml` | TOML file with `[system]`, `[network]`, `[sampler]`, `[sr]`, `[training]` |

**Output:** creates `{training.output_dir}/N{n_particles}_{YYYYmmdd-HHMMSS}/` containing `results.npz` and `config.toml`. `main()` also returns `(model, params)` when imported.

**Exit with error (`RuntimeError`)** if the thermalization drift is `|z| ≥ 3` or the acceptance is outside `[0.4, 0.6]`. See [usage.md](usage.md#common-errors).

## Conventions

- **Configuration** `X`: array `(N, dim)`. **Batch**: `(B, N, dim)`.
- **`log_psi`** is always a callable returning a complex scalar `log ψ = log|ψ| + iφ`:
  - `log_psi(x)`: θ fixed. Used by `boson_trap` and `MetroSampler`.
  - `log_psi(theta, x)`: θ is a flat real vector. Used by `SR`.
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
| `RBM` | `n_visible`, `n_hidden`, `init_scale` | `H` → real scalar, `a·H + Σ log cosh(b + W H)` |
| `FFNN` | `n_visible`, `n_hidden` | `H` → real scalar, `u · log cosh(V H + c)` |
| `NQS` | `n_visible`, `n_hidden_rbm`, `n_hidden_ffnn`, `alpha`, `init_scale=0.01` | `x (N, dim)` → complex `log ψ` |

Structure of `NQS` params:

```python
{
  "params_dse":  {"W": (F, 2), "b": (F,)},
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

## `src.sampler`

### `MetroSampler(n_chains, step_size=0.4, n_sweep=10)`

| Method | Signature | Returns |
|---|---|---|
| `init_walkers` | `(key, n_particles, dim)` | `(n_chains, N, dim)` drawn from `N(0, I)` |
| `step` | `(log_psi, (walkers, log_prob), key)` | `((walkers, log_prob), accept (n_chains,))`, one Metropolis step in `lax.scan` form |
| `sample` | `(log_psi, walkers, n_samples, key)` | `(samples (n_samples·n_chains, N, dim), walkers, mean_acceptance)` |
| `check_therm` | `(samples, z_max=3.0)` | `(thermalized: bool, z)` |

`sample` runs `n_samples` sweeps of `n_sweep` steps and records one configuration per chain per sweep. Sample order is sweep-major (`reshape` of `(n_samples, n_chains, …)`), and `check_therm` relies on that order.

```python
sampler = MetroSampler(n_chains=256)
walkers = sampler.init_walkers(key, 4, 2)
samples, walkers, acc = sampler.sample(lambda x: model.apply(params, x), walkers, 8, key2)
```

---

## `src.sr_optimizer`

| Function / class | Signature | Returns |
|---|---|---|
| `flatten_params` | `(params)` | `(theta (p,), unravel)` via `ravel_pytree` |
| `log_derivatives` | `(theta, log_psi, samples)` | `O (Ns, p)` complex, `O_k = ∂_θk log ψ` |
| `compute_S_F` | `(O, E_loc)` | `S (p, p)`, `F (p,)`, both complex covariances |
| `SR(learning_rate, varepsilon=1e-4)` | dataclass | — |
| `SR.step` | `(theta, log_psi, samples, E_loc)` | `(theta_new, [mean(Re E_loc), Var(E_loc)])` |

`SR.step` solves `(Re S + εI) δθ = −η Re F` and returns `θ + δθ`.

```python
theta, unravel = flatten_params(params)
log_psi = lambda t, x: model.apply(unravel(t), x)
sr = SR(learning_rate=0.05, varepsilon=1e-3)
theta, (E, var) = sr.step(theta, log_psi, samples, E_loc)
```
