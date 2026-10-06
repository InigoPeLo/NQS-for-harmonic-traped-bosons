# Technical decisions

## Plain JAX instead of a framework (NetKet / Flax)

The model, sampler and optimizer are hand-written as frozen dataclasses over raw `jax.numpy`. The only declared dependency is `jax[cuda12]`, and the code imports nothing else outside the standard library. Results are written with `jnp.savez`, which is NumPy's `savez` re-exported by JAX. NumPy is still installed, but only as a required dependency of JAX.

- **Why:** every equation in the notes maps to a few visible lines (S, F, local energy, the Metropolis criterion), which suits a project whose goal is to validate the method against an exact answer.
- **Trade-off:** no built-in samplers with adaptive steps, no `MCState` error estimates with autocorrelation times, and no iterative SR solvers, so these are either missing or implemented by hand (see below).

## Frozen dataclasses + explicit parameter pytrees

Each component is `@dataclass(frozen=True)`, and `init(key)` *returns* a parameter dict instead of storing it.

- Frozen dataclasses are immutable and hashable, so `make_train_step` can close over them and `jax.jit` treats them as static constants.
- The comment in `DSE.init` notes the alternative: storing parameters on `self` "wouldn't work". JAX transformations need parameters as explicit inputs to trace and differentiate.

## Flat parameter vector for SR

`flatten_params` (`ravel_pytree`) turns the nested dict into a single real vector θ plus `unravel`.

- SR needs `O` as an `(Ns, p)` matrix and S as `(p, p)`, which only makes sense over a flat vector.
- The model still works on readable nested params, and `log_psi = lambda t, x: model.apply(unravel(t), x)` is the only place where the two meet.
- **Cost:** the saved `theta` is meaningless without the same `NQS` config to rebuild `unravel`, which is why each run copies `config.toml` next to `results.npz`.

## Permutation symmetry by architecture (Deep Sets)

Sum pooling over a shared per-particle encoder makes ψ exactly symmetric.

- **Rejected-by-design alternative** (stated in the notes): an unconstrained ansatz. The Hamiltonian does not enforce particle statistics, so an unconstrained network could drift into states of the wrong symmetry.
- **Trade-off:** a single swish layer with sum pooling limits expressivity. That is enough here because the exact ground state is a product of Gaussians, but it is the first thing to grow when interactions are added.

## Gaussian envelope with `α = softplus(α̃)`

- Swish and `log cosh` grow at most linearly, so without `−α Σ|x|²` ψ would not be normalizable.
- SR is unconstrained and could push α below 0. The walkers would then drift to infinity. Reparametrizing through softplus guarantees α > 0, and the initial value is set by inverting softplus (`log(expm1(α))`).
- Starting from `α = 0.3` with RBM weights at scale `0.01` puts the initial state near a Gaussian, which makes thermalization and the first SR steps well behaved.

## Complex log ψ with real parameters

The network outputs `log|ψ| + iφ`, while θ is real.

- `jax.grad` needs real outputs, so every derivative is taken as `grad(Re) + i·grad(Im)`, both in `log_derivatives` and in `kinetic_local`.
- With real θ, the energy gradient is `2 Re F`, so only `Re S` and `Re F` enter the update. This halves the linear system compared to treating θ as complex.
- The phase network is redundant for this ground state (it is real and positive) but keeps the ansatz general.

## Numerical stability

- `log cosh(x)` is computed as `logaddexp(x, −x)`, which avoids `cosh` overflow. The dropped `−log 2` only rescales the norm or adds a global phase.
- The sampler works with `log|ψ|` throughout and compares `log u < 2Δ Re f`. The ratio `|ψ'/ψ|²` itself is never formed.
- JAX runs in its default float32 (`jax_enable_x64` is never set). This is enough to reach E = 4.0000 with N = 4. For relative errors below ~1e-6, enable x64.

## SR solver: dense solve with diagonal shift

`(Re S + εI) δθ = −η Re F` is solved with `jnp.linalg.solve`.

- With the defaults, `Ns = n_chains · n_samples = 256 · 8 = 2048 < p = 2273`. S has rank at most `Ns − 1`, so it is singular, and the shift `ε` (`varepsilon = 1e-3` in the config, class default `1e-4`) is required, not just a safeguard.
- S and F are sliced from one joint covariance of `(O, E_L)`. Computing S separately would repeat the `O(Ns·p²)` product for no benefit.
- **Trade-off:** forming S costs `O(Ns·p²)` and the solve costs `O(p³)`, both per iteration. This dominates the run time and grows with the cube of network width. Iterative (CG) or minSR (`Ns × Ns`) solvers would avoid it but are not implemented.

## Local energy via the full Hessian

The Laplacian is computed as `trace(jax.hessian(...))` over the `N·dim` coordinates, once for the real part and once for the imaginary part.

- Simple and exact.
- **Trade-off:** it builds a `(N·dim)²` matrix per sample. That is trivial for N = 4 (8 × 8 in 2D, 12 × 12 in 3D) but quadratic in N. A diagonal-only or forward-over-reverse Laplacian would scale better for larger systems.

## Sampler design

- **All-particle moves:** one Gaussian proposal moves every particle of a chain. It is simple and fully vectorized, but the acceptance rate drops as N grows at a fixed `step_size`.
- **Thinning:** only the last configuration of each `n_sweep`-step sweep is recorded, which trades compute for less correlated samples.
- **Walkers persist across iterations:** θ changes little per step, so the chains stay nearly in equilibrium and need no re-thermalization.
- **Fixed step size with hard gates:** instead of adapting `step_size`, `train.py` aborts if the thermalization acceptance is outside `[0.4, 0.6]` and tells you which way to adjust it. This keeps sampler behaviour explicit and reproducible.
- **Thermalization check:** `check_therm` compares the mean of `Σ|x_i|²` between the third and fourth quarters of the thermalization samples. Chains are independent, so the spread of per-chain drifts gives a standard error, and the run aborts if the drift exceeds 3σ.

## Final energy estimate

`train.py` reports the mean of the last 10% of iterations, with an error bar `std/√n_last`. As the code comment says, this ignores correlation between iterations and is only indicative. A rigorous estimate needs a long sampling run at fixed θ, which is not implemented.

## Config-to-constructor mapping

TOML sections are passed as `**kwargs` straight into the dataclasses. This removes glue code, and adding a field to a class makes it configurable immediately. The cost is that keys must match field names exactly: a typo raises `TypeError: unexpected keyword argument`.

The one exception is `dim`. It is a field of both `boson_trap` and `NQS`, but it is only read from `[system]` and injected into the model with `NQS(**cfg["network"], dim=system.dim)`. Keeping a single source means the sampler, the Hamiltonian and the encoder can never disagree on the particle dimension.

## Not a package

`[tool.uv] package = false`: uv only manages the environment, and code runs as scripts from the root with `src.*` imports. This avoids packaging boilerplate for research code, but it ties execution to the project root.
