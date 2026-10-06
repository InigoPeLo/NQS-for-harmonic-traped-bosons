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
│   ├── sampler.py        # Metropolis sampler
│   └── sr_optimizer.py   # Stochastic Reconfiguration
├── notebooks/
│   └── analysis.ipynb    # post-processing of results/ (stored without outputs)
├── notes/
│   └── NQS_TrappedBosons.pdf   # theory: Hamiltonian, ansatz, VMC, SR derivations
└── results/              # git-ignored; one subfolder per run
    └── N4_20261006-121907/      # current runs: N4_YYYYmmdd-HHMMSS-ffffff/
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
One module per component, and each one doesn't know about the others:

| Module | Contains | Depends on |
|---|---|---|
| `boson_trap.py` | `boson_trap`: `exact_energy`, `trap_potential`, `kinetic_local`, `local_energy`, `batch_local_energy` | a `log_psi(x)` callable |
| `nqs.py` | `DSE`, `RBM`, `FFNN` (building blocks), `NQS` (full model) | — |
| `sampler.py` | `MetroSampler`: `init_walkers`, `step`, `sample`, `check_therm` | a `log_psi(x)` callable |
| `sr_optimizer.py` | `flatten_params`, `log_derivatives`, `compute_S_F`, `SR.step` | a `log_psi(θ, x)` callable |

The modules communicate only through callables and arrays, never by importing each other. The system and sampler take `log_psi(x)` with θ fixed, while SR takes `log_psi(θ, x)` because it differentiates with respect to θ. Because of this, any ansatz that returns a complex `log ψ` can replace `NQS` without changing the other three modules.

### `config.toml`
The sections `[system]`, `[network]`, `[sampler]` and `[sr]` are unpacked with `**` into `boson_trap`, `NQS`, `MetroSampler` and `SR`. Their keys **must match the dataclass field names**. `[training]` holds loop-level settings that belong to no class.

### `notebooks/`
Post-processing only. `analysis.ipynb` reads every run in the folders of `RESULTS_DIRS` (default `results/`), lets you filter which ones to analyse, and never trains. It finds the project root by walking up to `pyproject.toml` and adds it to `sys.path`, so it imports `src.*` like `train.py` does. It can rebuild and sample a trained wavefunction because each run stores its own `config.toml`. It depends on the dev group (`matplotlib`, `ipykernel`), never on code in `train.py`.

### `notes/`
Reference material only, not read by the code. Comments in `src/nqs.py` cite its equations (eq. 58, eq. 64, section 6.2.1).

### `results/`
Generated output, git-ignored. Folder names encode `N` and the timestamp at save time down to microseconds (`N{N}_{YYYYmmdd-HHMMSS-ffffff}`), so runs launched in parallel don't collide. The four existing runs predate the microsecond suffix and are named `N4_YYYYmmdd-HHMMSS`.

## What should not go where

- **Mutable state in the component classes:** they are frozen dataclasses on purpose. Parameters, walkers and keys are passed in and returned. Storing them on `self` would break `jit` purity and hashing.
- **Hyperparameters hard-coded in `train.py`:** add them as a dataclass field and a matching key in the config section. The `**cfg[...]` unpacking picks them up automatically.
- **Analysis code in `train.py`:** it saves everything needed (`theta`, history, config). Post-processing belongs in `notebooks/`.
- **Run outputs in git:** `results/` is ignored. For the same reason, clear the notebook outputs before committing, since embedded figures make large, noisy diffs.
- **Plotting libraries in `[project] dependencies`:** they go in the dev group, so a training-only install (`uv sync --no-dev`) stays at JAX alone.
