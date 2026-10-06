# boson-trap

Neural quantum state (Deep Sets + RBM + FFNN) for N bosons in a 2D harmonic trap, trained with Variational Monte Carlo (VMC) and Stochastic Reconfiguration (SR), written in plain JAX.

## 📚 Documentation
| | |
|---|---|
| [⚙️ Architecture](docs/architecture.md) | System design, components and flow |
| [📁 Structure](docs/structure.md) | Project organization and responsibilities |
| [🚀 Installation](docs/installation.md) | Requirements and steps to run the project |
| [🧠 Technical decisions](docs/decisions.md) | Trade-offs and design justifications |
| [📖 Usage guide](docs/usage.md) | Running training, reading results, tuning, common errors |
| [🔌 API](docs/api.md) | CLI and Python interface of the `src` modules |

---

## Description

- **What it does:** finds the ground state of N non-interacting bosons in an isotropic 2D harmonic trap,
  $H = -\tfrac12\sum_i \nabla_i^2 + \tfrac12\omega^2\sum_i |x_i|^2$ (units ħ = m = 1), by minimizing the variational energy of a neural network wavefunction $\psi_\theta(x_1,\dots,x_N)$.
- **What problem it solves:** it is a testbed for a continuous-space NQS pipeline. The problem has an exact solution, $E_0 = N\,d\,\omega/2$ with $\psi_0 \propto e^{-\omega\sum_i|x_i|^2/2}$, so every part (the permutation-symmetric ansatz, the Metropolis sampler, the local energy with automatic derivatives, and SR) can be checked against a known answer.
- **Real use case:** validating the method before adding harder physics, such as interactions, where no closed-form answer exists. The theory and derivations are in [notes/NQS_TrappedBosons.pdf](notes/NQS_TrappedBosons.pdf).

## Quick start

```bash
uv sync
uv run python train.py --config config.toml
```

With the shipped `config.toml` (N = 4, 150 SR iterations), the energy converges to the exact value `E_0 = 4`:

```
Thermalization: z = 0.84
Thermalization: acceptance = 0.57
Iteration 1/150: E = 4.909703, Var(E) = 1.991309, acceptance = 0.56
...
Final energy (mean of the last 15 iterations): E = 4.000001 +- 0.000011
Exact energy: E_0 = 4.000000, relative error = 2.38e-07
Final Var(E_loc) = 3.14e-05, alpha = 0.4865 (exact 0.5)
Results saved in results/N4_20261006-121907
```

## Technologies used

| Category | Technology | Role |
|---|---|---|
| Language | Python ≥ 3.14 | Uses `tomllib` and frozen dataclasses |
| Numerics / autodiff | JAX (`jax[cuda12]` ≥ 0.11.2) | `jit`, `vmap`, `lax.scan`, `grad`, `hessian`, `ravel_pytree` |
| Hardware | NVIDIA GPU with CUDA 12 | Runs on the GPU when one is available, otherwise on the CPU |
| Environment | uv | Manages the environment and dependencies. The project is not installed as a package |
| Config | TOML | One section per component, unpacked straight into its constructor |
| Output | `.npz` via `jnp.savez` | Final parameters and per-iteration history |

## Quick installation

1. `git clone git@github.com:InigoPeLo/NQS-for-harmonic-traped-bosons.git && cd NQS-for-harmonic-traped-bosons`
2. `uv sync` (uv downloads Python 3.14 if it is missing)
3. `uv run python train.py`

Full details and troubleshooting: [docs/installation.md](docs/installation.md).

## Architecture (summary)

`train.py` builds four immutable components from `config.toml`:

- the Hamiltonian (`boson_trap`)
- the wavefunction (`NQS`)
- the Metropolis sampler (`MetroSampler`)
- the optimizer (`SR`)

The NQS maps each particle through a shared Deep Sets encoder and sums the results, so ψ is symmetric under particle exchange by construction. An RBM gives log|ψ|, a Gaussian envelope keeps ψ normalizable, and an FFNN gives the phase. Each SR iteration does three things, all inside one `jax.jit`-compiled step:

1. sample configurations from |ψ|²
2. compute local energies with automatic derivatives
3. solve `(Re S + εI) δθ = −η Re F`

See [docs/architecture.md](docs/architecture.md).

## Project structure

```
boson-trap/
├── train.py          # entry point: config → components → thermalize → SR loop → save
├── config.toml       # all run parameters, one section per component
├── src/
│   ├── boson_trap.py # Hamiltonian, local energy, exact energy
│   ├── nqs.py        # Deep Sets encoder + RBM + FFNN + Gaussian envelope
│   ├── sampler.py    # Metropolis sampler and thermalization check
│   └── sr_optimizer.py # log-derivatives, S and F, SR update
├── notes/            # theory notes (PDF)
└── results/          # one folder per run (git-ignored)
```

See [docs/structure.md](docs/structure.md).
