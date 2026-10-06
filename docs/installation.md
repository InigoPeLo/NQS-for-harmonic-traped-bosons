# Installation

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux x86_64 or aarch64 (including WSL2) | `jax[cuda12]` ships CUDA wheels only for Linux |
| Python | ≥ 3.14 | `requires-python = ">=3.14"`. uv can download it |
| [uv](https://docs.astral.sh/uv/) | recent | Manages the venv and dependencies |
| NVIDIA GPU + driver | CUDA 12 compatible | Optional. Without it JAX falls back to the CPU. CUDA libraries come as pip wheels, so no system CUDA toolkit is needed |

No database or external services.

## Steps

### 1. Clone

```bash
git clone git@github.com:InigoPeLo/NQS-for-harmonic-traped-bosons.git
cd NQS-for-harmonic-traped-bosons
```

### 2. Install dependencies

```bash
uv sync
```

This creates `.venv/` with `jax[cuda12]`, its CUDA 12 wheels and NumPy. `pyproject.toml` sets `[tool.uv] package = false`, so the project itself is **not** installed. Modules are imported as `src.nqs`, `src.sampler`, … which only works when commands run from the project root.

`uv.lock` and `.python-version` are committed. A fresh clone gets the same interpreter (3.14) and the exact package versions, including `jax`/`jaxlib` 0.11.2. To fail instead of silently re-resolving when the lockfile and `pyproject.toml` disagree:

```bash
uv sync --locked
```

### 3. Environment variables

None are required. Two optional JAX variables:

```bash
# Stop JAX from preallocating 75% of GPU memory. Avoids the
# "CUDA_ERROR_OUT_OF_MEMORY" log lines seen on small laptop GPUs / WSL2.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Force CPU (e.g. to compare against GPU or on a machine without CUDA)
export JAX_PLATFORMS=cpu
```

### 4. Run

```bash
uv run python train.py --config config.toml
```

`--config` defaults to `config.toml`, so `uv run python train.py` is equivalent.

## Verify it works

1. Check that JAX sees the GPU:

   ```bash
   uv run python -c "import jax; print(jax.__version__, jax.devices())"
   # 0.11.2 [CudaDevice(id=0)]      ← GPU
   # 0.11.2 [CpuDevice(id=0)]       ← CPU fallback
   ```

2. Run training with the shipped config. A healthy run passes both thermalization gates and ends near the exact energy:

   ```
   Thermalization: z = 0.84
   Thermalization: acceptance = 0.57
   ...
   Exact energy: E_0 = 4.000000, relative error = 2.38e-07
   Final Var(E_loc) = 3.14e-05, alpha = 0.4865 (exact 0.5)
   Results saved in results/N4_YYYYmmdd-HHMMSS
   ```

On an RTX 5070 Laptop GPU, 21 iterations take about 16 s including JIT compilation. The full 150-iteration default run scales from that.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `E... cuda_executor.cc ... CUDA_ERROR_OUT_OF_MEMORY` at startup, but training continues | JAX's default preallocation fails, then it retries with smaller blocks | Harmless. Silence it with `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| `ModuleNotFoundError: No module named 'src'` | Script run from outside the project root | `cd` to the root, then `uv run python train.py` |
| `FileNotFoundError: config.toml` | Same: the default `--config` is relative to the current directory | Run from the root or pass an absolute `--config` path |
| uv refuses to resolve / "requires-python" error | Python < 3.14 selected (e.g. `.python-version` overridden by `UV_PYTHON`) | `uv python install 3.14`, and unset `UV_PYTHON` if it is set |
