# Technical decisions

## Plain JAX instead of a framework (NetKet / Flax)

The model, sampler and optimizer are hand-written as frozen dataclasses over raw `jax.numpy`. The only declared dependency is `jax[cuda12]`, and the code imports nothing else outside the standard library. Results are written with `jnp.savez`, which is NumPy's `savez` re-exported by JAX. NumPy is still installed, but only as a required dependency of JAX.

- **Why:** every equation of the method maps to a few visible lines (S, F, local energy, the Metropolis criterion), which suits a project whose goal is to validate the method against an exact answer.
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

- **Rejected-by-design alternative:** an unconstrained ansatz. The Hamiltonian does not enforce particle statistics, so an unconstrained network could drift into states of the wrong symmetry.
- **Trade-off:** a single swish layer with sum pooling limits expressivity. That is enough here because the exact ground state is a product of Gaussians, but it is the first thing to grow when interactions are added.

## Gaussian envelope with `α = softplus(α̃)`

- Swish and `log cosh` grow at most linearly, so without `−α Σ|x|²` ψ would not be normalizable.
- SR is unconstrained and could push α below 0. The walkers would then drift to infinity. Reparametrizing through softplus guarantees α > 0, and the initial value is set by inverting softplus (`log(expm1(α))`).
- Starting from `α = 0.3` with RBM weights at scale `0.01` puts the initial state near a Gaussian, which makes thermalization and the first SR steps well behaved.

## The RBM models |ψ|², not |ψ|

The RBM output is `log|ψ|²` without the envelope, and `NQS.apply` returns `log|ψ| = ½·RBM − α Σ|x_i|²`.

- **Why:** with the hidden units traced out, `Σ_h exp(a·H + h·θ) ∝ exp(a·H + Σ log cosh θ)`. This makes the sampled distribution |ψ|² exactly the marginal of an RBM with **one** hidden layer, which is what `GibbsSampler` (alternating `p(h|x)` and `p(x|h)`) relies on. If the RBM modelled |ψ|, squaring it would need two copies of the hidden layer.
- **Only `NQS.apply` changes:** the model still returns `log ψ`. The Metropolis sampler (`2·Δ Re f` is `Δ log|ψ|²`), the local energy and SR are all written for `log ψ`, so none of them changes. Returning `log|ψ|²` instead would have spread factors of ½ and 2 across the sampler, the Laplacian, the log-derivatives and the phase.
- **Envelope kept in ψ:** α keeps its meaning (exact value `ω/2`) and its initial value in the config. In |ψ|² the envelope is `−2α Σ|x_i|²`.
- **Trade-off:** at a fixed number of hidden units M, `½ log cosh θ` and `log cosh θ` are not the same family of functions, though both are quadratic near 0 and linear far from it. An RBM over |ψ|² with 2M units contains the old |ψ| RBM with M units (pairs of tied units). The ½ also halves the RBM gradients, which changes how the absolute shift `ε` compares to S.
- **Compatibility:** a `theta` saved with the |ψ| RBM has the same shape but describes a different ψ under the current code. The saved config does not say which version produced a run.

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

- With the N = 4 settings, `Ns = n_chains · n_samples = 256 · 8 = 2048 < p = 2273`. S has rank at most `Ns − 1`, so it is singular, and the shift `ε` (`varepsilon = 1e-3` for N = 4, class default `1e-4`) is required, not just a safeguard.
- **ε is absolute, so it has to grow with N.** Sum pooling makes `H`, and with it every `O_k`, grow with N, so the scale of S grows roughly as N². The largest eigenvalue of `Re S` is 28 at N = 4 and 1.7·10⁵ at N = 40. There, `ε = 1e-3` no longer regularizes anything. `S + εI` has a condition number of ~10⁸, which is beyond float32's ~7 digits, and the update direction becomes noise:
  - the energy *rises* even with a small η;
  - after 2–3 steps the walkers fall behind ψ, `E_loc` goes below `E_0`, and the run turns NaN.

  The same run in float64 is stable, which confirms precision is the cause. The fix used is per-N config: `varepsilon = 1.0`, `learning_rate = 0.02` for N = 40. A shift relative to `diag(S)` was tried: it is stable at N = 40 but slows N = 4 convergence severalfold, so it was not adopted.
- S and F are sliced from one joint covariance of `(O, E_L)`. Computing S separately would repeat the `O(Ns·p²)` product for no benefit.
- **Trade-off:** forming S costs `O(Ns·p²)` and the solve costs `O(p³)`, both per iteration. This dominates the run time and grows with the cube of network width. Iterative (CG) or minSR (`Ns × Ns`) solvers would avoid it but are not implemented.

## Local energy via the full Hessian

The Laplacian is computed as `trace(jax.hessian(...))` over the `N·dim` coordinates, once for the real part and once for the imaginary part.

- Simple and exact.
- **Trade-off:** it builds a `(N·dim)²` matrix per sample. That is trivial for N = 4 (8 × 8 in 2D, 12 × 12 in 3D) but quadratic in N. A diagonal-only or forward-over-reverse Laplacian would scale better for larger systems.

## Sampler design

- **All-particle moves (Metropolis):** one Gaussian proposal moves every particle of a chain. It is simple and fully vectorized, but the acceptance rate drops as N grows at a fixed `step_size`. `GibbsSampler` moves one particle at a time instead (see below).
- **Thinning:** only the last configuration of each `n_sweep`-step sweep is recorded, which trades compute for less correlated samples.
- **Walkers persist across iterations:** θ changes little per step, so the chains stay nearly in equilibrium and need no re-thermalization.
- **Fixed step size, final acceptance warning:** `step_size` is never adapted. After training, `train.py` prints the mean acceptance of the last 20 iterations and a `Warning:` if it is outside `[ACC_MIN, ACC_MAX] = [0.25, 0.4]`. The band was lowered from `[0.4, 0.6]`: for random-walk moves in many dimensions the optimal acceptance is below 50% (it tends to ~0.23 as the dimension grows).
  - **Why not check after thermalization:** that check used to abort the run, but it was removed. The acceptance always *drops* during training, because ψ narrows as α grows (`|ψ|²` has width `1/(2√α)`), and it only settles once ψ has converged: 0.56 → 0.45 at N = 4, 0.51 → 0.34 at N = 40. A check at the start measures the initial ψ, not the one that matters, and its upper bound prevented choosing a step suited to the final ψ.
  - **Why not stop at a plateau:** a "plateau" detected from the acceptance alone gave false positives on slowly converging runs.
  - **Why a warning and not an error:** a low acceptance does not bias the energy, it only makes the samples more correlated. The N = 40 run converged to `E = 40.07` with an acceptance of 0.34.
- **Thermalization check:** shared by both samplers (`check_therm` is identical in `sampler_m.py` and `sampler_g.py`). It compares the mean of `Σ|x_i|²` between the third and fourth quarters of the thermalization samples. Chains are independent, so the spread of per-chain drifts gives a standard error, and the run aborts if the drift exceeds 3σ. With fewer than 4 samples per chain the quarters would be empty and `z` would be NaN, so `check_therm` raises a `ValueError` with an explicit message instead.

## Block Gibbs sampler as an alternative to Metropolis

`GibbsSampler` exploits two properties of the ansatz: the RBM over |ψ|² has one hidden layer, and the Deep Sets encoder is a sum over particles. Given the hidden units, `p(x | h)` factorizes into N identical one-body densities, so every particle is moved and accepted **on its own**.

- **Why:** with all-particle Metropolis moves, the proposal is `N·dim`-dimensional and `step_size` must shrink as N grows to keep the acceptance (0.4 at N = 4, 0.14 at N = 20, 0.15 with acceptance 0.34 at N = 40). Per-particle moves keep a fixed step meaningful at any N. The correlations between particles are carried by the exact `h | x` step instead of by small collective moves.
- **Metropolis inside Gibbs:** the one-body density `p_1(y | c)` has no closed-form sampler, so each Gibbs step makes `n_metro` random-walk moves that leave it invariant. The `h | x` step is exact.
- **Validation:** with the same parameters, Gibbs and Metropolis agree on `<Σ|x_i|²>` within error bars, also for RBM weights large enough to push the distribution far from the Gaussian envelope. Trained at N = 20 for 100 iterations with the shipped settings, both reach the same energy (`20.089` Gibbs, `20.096` Metropolis) in the same time (~19 s), since the local energy dominates the cost at that size.
- **Trade-off:** Gibbs is not ansatz-agnostic. It takes the parameter dict and re-implements |ψ|² from its pieces (`hidden_sample`, `log_p1`), so it duplicates the formula of `NQS.apply` and must be kept in sync with it by hand. A mismatch is not an error: it silently samples the wrong distribution. Metropolis only needs `log_psi(x)` and stays as the general option.
- **Acceptance is per particle,** so it is not directly comparable with the Metropolis one. The final acceptance warning of `train.py` uses the same `[0.25, 0.4]` band for both samplers; with Gibbs and `step_size = 0.5` the acceptance ends around 0.67 at N = 20 and triggers the "above 0.4" warning.

### `sample_fn`: one call signature for both samplers

`MetroSampler.sample` takes `log_psi(x)`, `GibbsSampler.sample` takes `params`. Instead of branching wherever sampling happens, `main()` builds `sample_fn(θ, walkers, n, key)` once, and thermalization and `train_step` only call it. Both samplers return the same `(samples, walkers, acceptance)` layout, so nothing downstream changes.

### Sampler settings in subsections

`[sampler]` only holds `type`, and each sampler reads its own subsection. `step_size` means different things in each (an `N·dim`-dimensional move for Metropolis, a single-particle move for Gibbs) and `n_metro` only exists in `GibbsSampler`, which would raise `TypeError` in `MetroSampler`. Subsections let both settings live in one file and switching sampler is a one-line change.

## Sampler comparison methodology (`evaluate/compare_samplers.py`)

Comparing two samplers is only meaningful if neither is handicapped by its settings. The script makes four choices for that:

- **Step sizes are tuned automatically, per N and per sampler,** by bisection in log scale until the acceptance is in `[ACC_MIN, ACC_MAX]`. A hand-tuned Metropolis against a mistuned Gibbs (or the reverse) would measure the settings, not the samplers. It tunes twice: at the initial θ, so that the reference training works, and at the trained θ, because that is the ψ whose acceptance matters (the same advice given for `train.py`).
- **The fixed-θ test records every step** (`n_sweep = 1`). Measured per recorded sweep, a sampler that thins a lot reaches τ = 1 and cannot show it is wasting work: an early version measured per sweep gave Gibbs τ = 1.00 and made it look 2× *less* efficient than Metropolis at N = 4. Per step, the same Gibbs is 2.4× *more* efficient.
- **The figure of merit is effective samples per second,** `n / (τ · t)`. τ alone ignores that a Gibbs step costs more; time per step alone ignores correlation. Their product is what decides how long a given error bar takes. Errors come from the spread of independent chain means, so they need no τ estimate; τ is only used for the efficiency.
- **The training comparison uses the config's `n_sweep`,** because that is how `train.py` would run. It reports convergence against iterations (sample quality as seen by SR) and against wall time (what it costs), separately.

**Reuse over copies:** the script imports `make_train_step` and the acceptance band from `train.py`, and `make_samplers` mirrors `sample_fn`. A comparison therefore runs exactly the training loop of a real run. The cost is that `train.py` must stay importable without side effects, which `if __name__ == "__main__"` guarantees.

**Estimator limits:** `autocorr_time` cuts the sum at the first non-positive ρ(k), which slightly underestimates τ (19 → 18.75 on an AR(1) test with φ = 0.9) equally for both samplers. Wall times depend on the GPU and include the per-step overhead of `lax.scan`, so they compare the samplers on one machine rather than in absolute terms.

**Outcome so far (N = 4, 20):** both samplers agree within 2σ; per step, Gibbs gives 2.4× and 8.6× more effective samples per second, since its τ does not grow with N; in training with `n_sweep = 10`, Metropolis still converges faster in wall time because Gibbs over-thins. The comparison therefore points to lowering `[sampler.gibbs] n_sweep` rather than to either sampler being better in general.

## Final energy estimate

`train.py` reports the mean of the last 10% of iterations, with an error bar `std/√n_last`. As the code comment says, this ignores correlation between iterations and is only indicative. The rigorous estimate, a long sampling run at fixed θ, lives in `notebooks/analysis.ipynb` (section 5), not in `train.py`:

- **Error bar from independent chains:** the error comes from the spread of per-chain means. Chains are independent, so this accounts for autocorrelation without a blocking analysis.
- **Training stays lean:** evaluation is optional and can be redone with more samples on any saved run without retraining, since each run stores `theta` and its config.

## Config-to-constructor mapping

TOML sections are passed as `**kwargs` straight into the dataclasses. This removes glue code, and adding a field to a class makes it configurable immediately. The cost is that keys must match field names exactly: a typo raises `TypeError: unexpected keyword argument`.

The sampler is unpacked from a subsection selected by `[sampler] type` (see above).

The other exception is `dim`. It is a field of both `boson_trap` and `NQS`, but it is only read from `[system]` and injected into the model with `NQS(**cfg["network"], dim=system.dim)`. Keeping a single source means the sampler, the Hamiltonian and the encoder can never disagree on the particle dimension.

## Not a package

`[tool.uv] package = false`: uv only manages the environment, and code runs as scripts from the root with `src.*` imports. This avoids packaging boilerplate for research code, but it ties execution to the project root.
