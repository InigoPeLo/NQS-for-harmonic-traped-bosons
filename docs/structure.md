# Structure

```
boson-trap/
├── train.py              # entry point (CLI)
├── config.toml           # run parameters
├── pyproject.toml        # metadata + dependencies (+ dev group); uv in non-package mode
├── .python-version       # 3.14; uv picks this interpreter
├── uv.lock               # exact resolved versions (jax/jaxlib 0.11.2, numpy 2.5.3, CUDA wheels)
├── src/
│   ├── __init__.py       # empty; makes `src` importable as a package
│   ├── boson_trap.py     # physical system
│   ├── nqs.py            # neural wavefunction
│   ├── sampler_m.py      # Metropolis sampler
│   ├── sampler_g.py      # block Gibbs sampler
│   └── sr_optimizer.py   # Stochastic Reconfiguration
├── evaluate/
│   ├── compare_samplers.py   # Metropolis vs Gibbs comparison (CLI)
│   └── output/           # git-ignored; one subfolder per comparison
│       └── compare_N4-20_20261009-111055/   # compare_N{N1-N2-…}_YYYYmmdd-HHMMSS/
│           ├── summary.json    # tuned step sizes, training summaries, fixed-θ measurements per N
│           ├── histories.npz   # training curves, keys N{N}_{sampler}_{field}
│           └── config.toml     # copy of the config used (before tuning)
├── notebooks/
│   ├── analysis.ipynb    # post-processing of results/ (stored without outputs)
│   └── compare_samplers.ipynb  # post-processing of evaluate/output/
├── notes/                # git-ignored; local theory notes, not in the repository
│   ├── NQS_TrappedBosons.pdf   # method notes (equation numbers cited in the code)
│   └── gibbs_sampler.tex/.pdf  # derivation of the Gibbs sampler
└── results/              # git-ignored; one subfolder per run
    └── N4_20261008-133049-200873_g/  # N{N}_YYYYmmdd-HHMMSS-ffffff_{m|g}/
        ├── config.toml   # copy of the config used
        └── results.npz   # theta + training history
```

## Responsibilities

### `train.py`
Orchestration only:

- parse `--config`
- build the components
- seed and split PRNG keys
- thermalize the chains and gate on the checks
- run the jitted SR loop
- print diagnostics and save the run

It holds no physics or numerics beyond averaging the last 10% of iterations for the final estimate.

### `src/`
One module per component:

| Module | Contains | Depends on |
|---|---|---|
| `boson_trap.py` | `boson_trap`: `exact_energy`, `trap_potential`, `kinetic_local`, `local_energy`, `batch_local_energy` | a `log_psi(x)` callable |
| `nqs.py` | `DSE`, `RBM`, `FFNN` (building blocks), `NQS` (full model) | — |
| `sampler_m.py` | `MetroSampler`: `init_walkers`, `step`, `sample`, `check_therm` | a `log_psi(x)` callable |
| `sampler_g.py` | `GibbsSampler`: `init_walkers`, `hidden_sample`, `log_p1`, `metro_step`, `step`, `sample`, `check_therm` | the `NQS` parameter dict, and `DSE` imported from `nqs.py` |
| `sr_optimizer.py` | `flatten_params`, `log_derivatives`, `compute_S_F`, `SR.step` | a `log_psi(θ, x)` callable |

`boson_trap`, `MetroSampler` and `SR` communicate only through callables and arrays. The system and Metropolis take `log_psi(x)` with θ fixed, while SR takes `log_psi(θ, x)` because it differentiates with respect to θ. Any ansatz that returns a complex `log ψ` can replace `NQS` without changing them.

`sampler_g.py` is the exception: it is **tied to this `NQS`**. It imports `DSE`, reads the keys of the parameter dict (`params_dse`, `params_rbm`, `alpha_tilde`) and re-implements |ψ|² term by term in `hidden_sample` and `log_p1`. A change to the encoder, the RBM, the ½ factor or the envelope in `nqs.py` has to be mirrored there, or Gibbs silently samples a different distribution. `train.py` still only depends on the common `sample` output, through `sample_fn`.

### `config.toml`
The sections `[system]`, `[network]` and `[sr]` are unpacked with `**` into `boson_trap`, `NQS` and `SR`. `[sampler]` only holds `type` (`"metropolis"` or `"gibbs"`), and the matching subsection, `[sampler.metropolis]` or `[sampler.gibbs]`, is unpacked into `MetroSampler` or `GibbsSampler`. Keys **must match the dataclass field names**. `[training]` holds loop-level settings that belong to no class.

### `evaluate/`
Scripts that measure the method rather than produce a ground state. `compare_samplers.py` compares the two samplers for a list of N: it tunes their step sizes, trains with each one and samples a fixed θ with both. Unlike the notebook, it **does** train, and it reuses `train.py` instead of duplicating it: it imports `make_train_step`, `SAMPLER_TAGS`, `ACC_MIN` and `ACC_MAX`, so a comparison always runs the same SR iteration and the same acceptance band as a real training. Like the notebook, it finds the project root by walking up to `pyproject.toml`, so it runs from any directory. Its output goes to `evaluate/output/`, separate from the runs of `results/`.

### `notebooks/`
Post-processing only. `analysis.ipynb` reads every run in the folders of `RESULTS_DIRS` (default `results/`), lets you filter which ones to analyse, and never trains. `compare_samplers.ipynb` reads a comparison folder of `evaluate/output/` instead. They are separate notebooks so that each one runs with *Run All* on its own: `analysis.ipynb` stops with an error when `results/` is empty, which would otherwise block the comparison. `compare_samplers.ipynb` only reads `summary.json` and `histories.npz`, so it imports neither JAX nor `src`. It finds the project root by walking up to `pyproject.toml` and adds it to `sys.path`, so it imports `src.*` like `train.py` does. It can rebuild and sample a trained wavefunction because each run stores its own `config.toml`. It depends on the dev group (`matplotlib`, `ipykernel`), never on code in `train.py`.

### `notes/`
Git-ignored: the author's theory notes are kept locally and are not distributed with the repository. Nothing in the code reads them. The equation numbers cited in comments in `src/nqs.py` (eq. 58, eq. 64, section 6.2.1) refer to `NQS_TrappedBosons.pdf`. `gibbs_sampler.tex` derives the joint distribution, the two conditionals and the Metropolis-within-Gibbs step implemented in `sampler_g.py`.

### `results/`
Generated output, git-ignored. Folder names encode `N`, the timestamp at save time down to microseconds, and the sampler (`N{N}_{YYYYmmdd-HHMMSS-ffffff}_{m|g}`, `_m` for Metropolis and `_g` for Gibbs), so runs launched in parallel don't collide. The tag goes last so the names still sort by date. Older runs are named `N{N}_YYYYmmdd-HHMMSS-ffffff` (before the sampler tag, always Metropolis) or `N{N}_YYYYmmdd-HHMMSS` (before the microsecond suffix). The notebook accepts all three forms, and reads the sampler from the run's `config.toml`, not from the name.

## What should not go where

- **Sampler-specific branches in `train_step`:** the difference between samplers is confined to `sample_fn` in `main()`. The rest of the loop must not depend on which sampler is used.
- **Mutable state in the component classes:** they are frozen dataclasses on purpose. Parameters, walkers and keys are passed in and returned. Storing them on `self` would break `jit` purity and hashing.
- **Hyperparameters hard-coded in `train.py`:** add them as a dataclass field and a matching key in the config section. The `**cfg[...]` unpacking picks them up automatically.
- **Analysis code in `train.py`:** it saves everything needed (`theta`, history, config). Post-processing belongs in `notebooks/`, and experiments that need their own trainings or sampling runs (like the sampler comparison) in `evaluate/`. Plots of those experiments still go in the notebook, which reads their saved output.
- **Run outputs in git:** `results/` and `evaluate/output/` are ignored. For the same reason, clear the notebook outputs before committing, since embedded figures make large, noisy diffs.
- **Plotting libraries in `[project] dependencies`:** they go in the dev group, so a training-only install (`uv sync --no-dev`) stays at JAX alone.
