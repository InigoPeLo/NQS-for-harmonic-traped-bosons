# Architecture

The project is a single-script VMC pipeline. Four immutable components, each a `@dataclass(frozen=True)` in `src/`, are built from `config.toml` and wired together by `train.py`:

| Component | Class | Responsibility |
|---|---|---|
| Physical system | `boson_trap` (`src/boson_trap.py`) | Potential, local kinetic and total energy, exact reference energy |
| Wavefunction | `NQS` (`src/nqs.py`) | `log ψ_θ(X)` as a complex number, from a parameter pytree |
| Sampler | `MetroSampler` (`src/sampler.py`) | Draws configurations from \|ψ\|², checks thermalization |
| Optimizer | `SR` (`src/sr_optimizer.py`) | Log-derivatives, S matrix and force F, parameter update |

Components never hold parameters or state. Parameters live in a pytree (and later a flat vector `θ`), walkers in an array, and both are passed explicitly. This is what lets the whole iteration be one pure function under `jax.jit`.

## Wavefunction ansatz

A configuration is `X` of shape `(N, dim)`. `dim` is set once in `[system]` (default 2), and `train.py` passes `system.dim` to `NQS`, which forwards it to the encoder. The network returns $f_\theta(X) = \log\psi_\theta(X)$:

```
x_i ──► Deep Sets encoder ──► H = Σ_i swish(W x_i + b)        (F-dim, permutation invariant)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          RBM (amplitude)                  FFNN (phase)
  a·H + Σ_m log cosh(b_m + W_m·H)    Σ_k u_k log cosh(c_k + V_k·H)
                    │                             │
                    ▼                             ▼
        log|ψ| = RBM − α Σ_i |x_i|²          φ (phase)
                    └──────────► f = log|ψ| + i φ
```

- **Deep Sets encoder (`DSE`):** the same single-layer map is applied to every particle and the results are **summed**. Any permutation of particles gives the same `H`, so ψ is bosonic by construction and needs no explicit symmetrization.
- **RBM:** the log of an RBM with its hidden units traced out, applied to the continuous latent vector `H`. `log cosh` is computed as `logaddexp(θ, −θ)`, dropping the constant `−log 2`, to avoid overflow.
- **Gaussian envelope:** the RBM and swish grow at most linearly in |x|, so the factor `−α Σ|x_i|²` is what makes ψ decay. α is stored as `alpha_tilde` with `α = softplus(alpha_tilde)`, so SR can never make it negative.
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

`MetroSampler` runs `n_chains` independent random-walk Metropolis chains in parallel:

- proposal `X' = X + δ·ξ`, `ξ ~ N(0, I)`, moving **all** particles of a chain at once
- accept if `log u < 2·(Re f(X') − Re f(X))`, i.e. with probability `min(1, |ψ'|²/|ψ|²)`
- `n_sweep` steps form a sweep, and only the configuration at the end of each sweep is recorded. This thins the chain to reduce autocorrelation.
- both loops (steps in a sweep, sweeps in a call) are `jax.lax.scan`, so they compile into one XLA loop

`sample` returns `(n_samples·n_chains, N, dim)` samples, the final walkers, and the mean acceptance rate.

## Optimization: Stochastic Reconfiguration

With $O_k(X) = \partial_{\theta_k}\log\psi_\theta(X)$, computed per sample as `grad(Re f) + i·grad(Im f)`:

$$S_{kl} = \langle O_k^* O_l\rangle - \langle O_k^*\rangle\langle O_l\rangle,\qquad F_k = \langle O_k^* E_L\rangle - \langle O_k^*\rangle\langle E_L\rangle$$

Both come from a single `jnp.cov` call (with `bias=True`) on the joint variables `(O_1, …, O_p, E_L)`. `S` is the top-left `p × p` block, and `F` is the last column without the `Var(E_L)` corner. The `O(Ns·p²)` product is formed once per iteration. Since θ is real, only the real parts are used:

$$(\mathrm{Re}\,S + \varepsilon I)\,\delta\theta = -\eta\,\mathrm{Re}\,F,\qquad \theta \leftarrow \theta + \delta\theta$$

solved with a dense `jnp.linalg.solve`. `SR.step` also returns `[mean(E_L), var(E_L)]`. As θ approaches an eigenstate, `var(E_L)` goes to 0 (the zero-variance property), which makes it a convergence check independent of the energy. `jnp.var` is taken over the complex `E_L`, so the reported value is `Var(Re E_L) + Var(Im E_L)`. The imaginary part is not constant unless the learned phase is exactly flat.

## Data flow of a run

```
config.toml ─► load_config ─► boson_trap / NQS / MetroSampler / SR
                                        │
PRNGKey(seed) ─split─► key_params ─► NQS.init ─► params ─ravel_pytree─► θ, unravel
                    ├─► key_walkers ─► init_walkers ─► walkers (n_chains, N, 2) ~ N(0, I)
                    ├─► key_therm ──► sample(n_thermalization sweeps) ─► check_therm (|z| < 3)
                    └─► key_train ──► for n in n_iter:
                                         train_step(θ, walkers, key)   ← jax.jit
                                           ├ sample(n_samples sweeps)   walkers carried over
                                           ├ batch_local_energy         E_L (Ns,)
                                           └ SR.step                    θ_new, [E, Var]
                                       ─► summary + final acceptance warning (last 20 iterations)
                                       ─► save_results ─► results/N{N}_{YYYYmmdd-HHMMSS-ffffff}/
```

- `log_psi(θ, x) = model.apply(unravel(θ), x)` is the single bridge between the pytree world (model) and the flat vector world (SR).
- `make_train_step` closes over `log_psi`, `sampler`, `system`, `sr` and `n_samples`. `jit` treats them as compile-time constants, so `train_step` only takes the arrays that change: `θ`, `walkers` and `key`.
- Walkers are carried across iterations. Each iteration starts from an almost-equilibrated state, so it only needs `n_samples` sweeps and no re-thermalization.

## Persistence

There is no database. Each run writes:

- `results.npz` with `theta` and the per-iteration arrays `energy`, `variance`, `acceptance`, `alpha`
- `config.toml`, a verbatim copy of the input config

The config copy is needed to rebuild the same `NQS` (and so the same `unravel`) when reading `theta` back. That includes `[system].dim`, which sets the shape of the encoder weights.
