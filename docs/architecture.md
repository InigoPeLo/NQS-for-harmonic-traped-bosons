# Architecture

The project is a single-script VMC pipeline. Four immutable components, each a `@dataclass(frozen=True)` in `src/`, are built from `config.toml` and wired together by `train.py`:

| Component | Class | Responsibility |
|---|---|---|
| Physical system | `boson_trap` (`src/boson_trap.py`) | Potential, local kinetic and total energy, exact reference energy |
| Wavefunction | `NQS` (`src/nqs.py`) | `log ψ_θ(X)` as a complex number, from a parameter pytree |
| Sampler | `MetroSampler` (`src/sampler_m.py`) or `GibbsSampler` (`src/sampler_g.py`), chosen by `[sampler] type` | Draws configurations from \|ψ\|², checks thermalization |
| Optimizer | `SR` (`src/sr_optimizer.py`) | Log-derivatives, S matrix and force F, parameter update |

Components never hold parameters or state. Parameters live in a pytree (and later a flat vector `θ`), walkers in an array, and both are passed explicitly. This is what lets the whole iteration be one pure function under `jax.jit`.

## Wavefunction ansatz

A configuration is `X` of shape `(N, dim)`. `dim` is set once in `[system]` (default 2), and `train.py` passes `system.dim` to `NQS`, which forwards it to the encoder. The network returns $f_\theta(X) = \log\psi_\theta(X)$:

```
x_i ──► Deep Sets encoder ──► H = Σ_i swish(W x_i + b)        (F-dim, permutation invariant)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          RBM (probability)                FFNN (phase)
  a·H + Σ_m log cosh(b_m + W_m·H)    Σ_k u_k log cosh(c_k + V_k·H)
                    │                             │
                    ▼                             ▼
      log|ψ| = ½·RBM − α Σ_i |x_i|²          φ (phase)
                    └──────────► f = log|ψ| + i φ
```

- **Deep Sets encoder (`DSE`):** the same single-layer map is applied to every particle and the results are **summed**. Any permutation of particles gives the same `H`, so ψ is bosonic by construction and needs no explicit symmetrization.
- **RBM:** the log of an RBM with its hidden units traced out, applied to the continuous latent vector `H`. Its output models the probability density, `log|ψ|² = RBM − 2α Σ|x_i|²`, so `NQS.apply` multiplies it by ½ to get `log|ψ|`. `log cosh` is computed as `logaddexp(θ, −θ)`, dropping the constant `−log 2`, to avoid overflow.
- **Gaussian envelope:** the RBM and swish grow at most linearly in |x|, so the factor `−α Σ|x_i|²` is what makes ψ decay. It is applied to `log|ψ|`, so it is `−2α Σ|x_i|²` in `|ψ|²` and the exact value stays `α = ω/2`. α is stored as `alpha_tilde` with `α = softplus(alpha_tilde)`, so SR can never make it negative.
- **FFNN phase:** one hidden layer with `log cosh` activation. The exact ground state is real and positive, so the phase should learn to be constant.

Parameter count for `F` = `n_visible`, `M` = `n_hidden_rbm`, `K` = `n_hidden_ffnn`:

```
p = F·(dim+1)  +  M·(F+1) + F  +  K·(F+2)  +  1
    encoder       RBM            FFNN         alpha_tilde
```

The default `F = M = K = 32` gives `p = 2273` in 2D and `p = 2305` in 3D. Only the encoder weights depend on `dim`.

At initialization `a = b = 0` and `W_RBM ~ init_scale·N(0,1)` with `init_scale = 0.01`, so the RBM is almost constant. The starting state is therefore close to a pure Gaussian with `α = 0.3`, a better starting point than random weights. The exact answer is `α = ω/2 = 0.5`.

## Local energy

`boson_trap.local_energy` evaluates $E_L(X) = T_L(X) + V(X)$, where $V = \tfrac12\omega^2\sum_i|x_i|^2$ and

$$T_L = -\tfrac12\left[\nabla^2 f + (\nabla f)\cdot(\nabla f)\right],\qquad f = u + i\varphi .$$

`jax.grad` and `jax.hessian` need real outputs, so the real part `u` and imaginary part `φ` of `log ψ` are differentiated separately over the flattened `(N·dim,)` coordinates. The results are recombined as `∇²u + i∇²φ + (|∇u|² − |∇φ|²) + 2i ∇u·∇φ`. The Laplacian is the trace of the full Hessian. `batch_local_energy` uses `vmap` over samples.

## Sampling

Two samplers draw configurations from the same distribution, |ψ|². `[sampler] type` in the config selects one, and each reads its own subsection (`[sampler.metropolis]` or `[sampler.gibbs]`). Both run `n_chains` independent chains in parallel and have the same `init_walkers`, `sample` output and `check_therm`.

### `MetroSampler`: random-walk Metropolis

- proposal `X' = X + δ·ξ`, `ξ ~ N(0, I)`, moving **all** particles of a chain at once
- accept if `log u < 2·(Re f(X') − Re f(X))`, i.e. with probability `min(1, |ψ'|²/|ψ|²)`
- it only evaluates `log_psi(x)`, so it works with any ansatz
- `n_sweep` steps form a sweep, and only the configuration at the end of each sweep is recorded. This thins the chain to reduce autocorrelation.
- both loops (steps in a sweep, sweeps in a call) are `jax.lax.scan`, so they compile into one XLA loop

### `GibbsSampler`: block Gibbs over the RBM hidden units

Tracing out the hidden units of the RBM gives `log cosh`, so |ψ|² is exactly the marginal in `x` of the joint distribution

$$p(x, h) \propto \exp\Big[a\cdot H(x) + \sum_j h_j\,\theta_j(x) - 2\alpha\sum_i |x_i|^2\Big],\qquad h_j = \pm1,\quad \theta = b + W H(x).$$

Each Gibbs `step` alternates the two conditionals:

1. **`hidden_sample`, h | x (exact):** the hidden units are independent given `x`, with `p(h_j = +1 | x) = sigmoid(2θ_j)`.
2. **x | h (Metropolis inside Gibbs):** with `c = a + h·W` of shape `(n_chains, F)`, the exponent becomes `Σ_i [c·swish(W_d x_i + b_d) − 2α|x_i|²]`. Because `H` is a sum over particles, `p(x | h)` **factorizes over particles**, which all follow the same one-body density `p_1(y | c)` (`log_p1`). It has no closed-form sampler, so `metro_step` makes `n_metro` Gaussian random-walk moves of every particle, accepting or rejecting **each particle on its own**.

All correlations between particles go through `h`. `log_p1` is recomputed at the start of every Gibbs step, because the previous value was computed with a different `c`. Sweeps and recording work as in `MetroSampler`: `n_sweep` Gibbs steps per recorded sample, nested `lax.scan` loops. The reported acceptance is per particle.

`GibbsSampler` never calls `NQS.apply` and never evaluates the phase, which does not enter |ψ|². It takes the parameter dict (`unravel(θ)`) and rebuilds the encoder, the RBM terms and α itself.

### Common output

`sample` returns `(n_samples·n_chains, N, dim)` samples in sweep-major order, the final walkers, and the mean acceptance rate. `check_therm` relies on that order.

## Optimization: Stochastic Reconfiguration

With $O_k(X) = \partial_{\theta_k}\log\psi_\theta(X)$, computed per sample as `grad(Re f) + i·grad(Im f)`:

$$S_{kl} = \langle O_k^* O_l\rangle - \langle O_k^*\rangle\langle O_l\rangle,\qquad F_k = \langle O_k^* E_L\rangle - \langle O_k^*\rangle\langle E_L\rangle$$

Both come from a single `jnp.cov` call (with `bias=True`) on the joint variables `(O_1, …, O_p, E_L)`. `S` is the top-left `p × p` block, and `F` is the last column without the `Var(E_L)` corner. The `O(Ns·p²)` product is formed once per iteration. Since θ is real, only the real parts are used:

$$(\mathrm{Re}\,S + \varepsilon I)\,\delta\theta = -\eta\,\mathrm{Re}\,F,\qquad \theta \leftarrow \theta + \delta\theta$$

solved with a dense `jnp.linalg.solve`. `SR.step` also returns `[mean(E_L), var(E_L)]`. As θ approaches an eigenstate, `var(E_L)` goes to 0 (the zero-variance property), which makes it a convergence check independent of the energy. `jnp.var` is taken over the complex `E_L`, so the reported value is `Var(Re E_L) + Var(Im E_L)`. The imaginary part is not constant unless the learned phase is exactly flat.

## Data flow of a run

```
config.toml ─► load_config ─► boson_trap / NQS / MetroSampler or GibbsSampler / SR
                                        │
PRNGKey(seed) ─split─► key_params ─► NQS.init ─► params ─ravel_pytree─► θ, unravel
                    ├─► key_walkers ─► init_walkers ─► walkers (n_chains, N, 2) ~ N(0, I)
                    ├─► key_therm ──► sample_fn(n_thermalization sweeps) ─► check_therm (|z| < 3)
                    └─► key_train ──► for n in n_iter:
                                         train_step(θ, walkers, key)   ← jax.jit
                                           ├ sample_fn(n_samples sweeps) walkers carried over
                                           ├ batch_local_energy         E_L (Ns,)
                                           ├ SR.step                    θ_new, [E, Var]
                                           └ std(Im log ψ) over samples phase_std
                                       ─► summary + final acceptance warning (last 20 iterations)
                                       ─► save_results ─► results/N{N}_{YYYYmmdd-HHMMSS-ffffff}_{m|g}/
```

- `log_psi(θ, x) = model.apply(unravel(θ), x)` is the single bridge between the pytree world (model) and the flat vector world (SR).
- `sample_fn(θ, walkers, n, key)` hides which sampler is used. The two `sample` methods take different inputs: Metropolis needs `log_psi(x)`, Gibbs needs the parameter dict. `main()` builds `sample_fn` once, passing `lambda x: log_psi(θ, x)` or `unravel(θ)`, and thermalization and `train_step` only ever call `sample_fn`.
- `make_train_step` closes over `log_psi`, `sample_fn`, `system`, `sr` and `n_samples`. `jit` treats them as compile-time constants, so `train_step` only takes the arrays that change: `θ`, `walkers` and `key`.
- Walkers are carried across iterations. Each iteration starts from an almost-equilibrated state, so it only needs `n_samples` sweeps and no re-thermalization.

## Persistence

There is no database. Each run writes:

- `results.npz` with `theta` and the per-iteration arrays `energy`, `variance`, `acceptance`, `alpha`, `phase_std`. `phase_std` is the std of the phase `Im log ψ` over the samples of each iteration. The exact ground state has a constant phase, so it should go to 0.
- `config.toml`, a verbatim copy of the input config

The config copy is needed to rebuild the same `NQS` (and so the same `unravel`) when reading `theta` back. That includes `[system].dim`, which sets the shape of the encoder weights.

Runs saved before the Gibbs sampler was added have a flat `[sampler]` section (`n_chains`, `step_size`, `n_sweep`) and always used Metropolis. Current configs have `[sampler] type` plus one subsection per sampler.

The config does not record whether the RBM modelled |ψ| (older code) or |ψ|² (current code). A `theta` saved before that change has the same shape but gives a different ψ with the current `NQS`, so it must be read with the code that produced it.

## Sampler comparison (`evaluate/compare_samplers.py`)

A separate pipeline that measures the samplers instead of training a final state. For every N in `--N`:

```
config.toml (n_particles replaced by N)
   │
   ├─ setup ─────────────► system, NQS, initial θ, log_psi(θ, x), unravel, both samplers
   ├─ tune_samplers(θ_init) ─► step_size of each sampler with acceptance in [ACC_MIN, ACC_MAX]
   ├─ train_theta(metropolis) ─► reference θ_ref (n_iter SR iterations)
   ├─ tune_samplers(θ_ref) ──► step sizes for the trained ψ (the acceptance changes while ψ narrows or widens)
   ├─ compare_training ─────► both samplers train from the same initial θ with the same key:
   │                           E, Var, acceptance and wall time per iteration
   └─ compare_fixed_theta(θ_ref) ─► both samplers sample θ_ref recording every step (n_sweep = 1):
                               <Σ|x_i|²>, <E_loc> ± error, τ per step, effective samples per second
─► save_results ─► evaluate/output/compare_N{…}_{YYYYmmdd-HHMMSS}/
```

- **Same code as training:** `train_theta` thermalizes with `check_therm` and runs `make_train_step` from `train.py`, with the `sample_fn` of the chosen sampler. `make_samplers` builds the same `sample_fn` as `train.py` for both samplers at once.
- **Step tuning:** `tune_step_size` bisects `step_size` in log scale (double or halve until the band is bracketed, then the geometric mean), measuring the acceptance on 20 sweeps per trial. The walkers are thermalized once and carried between trials, because |ψ|² does not depend on the step.
- **Statistics:** `autocorr_time` computes the integrated autocorrelation time `τ = 1 + 2 Σ_k ρ(k)` of an `(n_times, n_chains)` series, averaging ρ over the chains and cutting the sum at the first non-positive ρ. `mean_and_error` takes the error bar from the spread of the chain means. `Σ|x_i|²` is measured on every step; `E_loc`, which needs a Hessian per sample, every `e_stride` steps and in batches.
- **Figure of merit:** `ESS/s = n_samples / (τ · wall time)`, the number of independent samples per second. Sampling is compiled ahead of time (`jit(...).lower(...).compile()`) and timed with `block_until_ready`, so compilation is not counted.
