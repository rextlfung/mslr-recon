# mslr-recon

**Multi-scale Low-Rank (MSLR) Reconstruction in Julia**

Iterative reconstruction of 3D + time MRI data. Uses a SENSE forward model and
a **multi-scale low-rank decomposition** regulariser optimised with POGM
(Proximal Optimised Gradient Method). Supports both CPU (multi-threaded) and 
GPU (CUDA) execution.

---

## Background

Accelerated MRI acquisitions collect only a fraction of k-space at each time
frame. Recovering a full image time series from this undersampled data requires
exploiting the structure of the signal — in this case, the fact that nearby
voxels tend to have correlated temporal dynamics (locally low-rank structure).

This codebase implements the **multiscale low-rank matrix decomposition** of
[Ong & Lustig (2016)](#references), which represents the image as a sum of
components each regularised at a different spatial scale:

```
X_final = X_global + X_regional + X_local + ...
```

Each component independently captures low-rank temporal structure at its own
patch size, from a single global component spanning the whole volume down to
small local patches. Data consistency is enforced on the sum. This is more
expressive than applying priors sequentially — the scales cannot interfere
with each other.

---

## Algorithm

The reconstruction solves:

$$\min_{\mathbf{X}} \; \frac{1}{2} \left\| \mathcal{A}\!\left(\sum_k \mathbf{X}_k\right) - \mathbf{Y} \right\|_F^2 \;+\; \sum_k \lambda_k \left\| \mathcal{P}_k(\mathbf{X}_k) \right\|_*$$

where:
- $\mathcal{A}$ is the block-diagonal SENSE encoding operator (one block per time frame)
- $\mathbf{Y}$ is the measured k-space data
- $\mathcal{P}_k(\mathbf{X}_k)$ reshapes spatial patches of component $k$ as **(voxels × time)** matrices
- $\|\cdot\|_*$ is the nuclear norm, promoting temporal low-rank structure within each patch
- $\lambda_k$ is set automatically via the Ong & Lustig (2016) formula — no manual tuning needed:

$$\lambda_k = \sqrt{p_k} + \sqrt{N_t} + \sqrt{\log\!\left(\frac{N_{vox} \cdot N_t}{\max(p_k,\, N_t)}\right)}$$

where $p_k$ is the number of voxels in a patch at scale $k$.

Optimisation uses **POGM** (Proximal Optimised Gradient Method) with gradient
restart. The Lipschitz constant is $L = N_{scales} \cdot \sigma_1(\mathcal{A})^2$.

---

## Repository structure

```
mslr-recon/
├── Project.toml              # Julia package dependencies
│
├── src/
│   ├── recon.jl              # Patch extraction/recombination, SVST, k-space utilities
│   ├── mirt_mod.jl           # POGM with restart, power iteration (modified from MIRT)
│   ├── analysis.jl           # tSNR maps, convergence plots, detrending
│   └── sense_gpu.jl          # GPU-native SENSE operator and patch SVD (requires CUDA.jl)
│
├── scripts/
│   ├── reconstruct.jl        # Reconstruction module — called by experiment files
│   └── analyze.jl            # Post-reconstruction analysis and visualisation
│
└── experiments/
    ├── 20251106_balltap.jl   # Ball phantom + finger-tapping, 18 coils, Nt=300
    ├── 20241017_fingertap.jl # Finger-tapping, 10 coils, Nt=300
    └── 20260317tap.jl        # Finger-tapping, 18 coils, Nt=387, half-overlapping patches
```

---

## Getting started

### Requirements

- Julia ≥ 1.9 (tested on 1.12)
- [BART](https://mrirecon.github.io/bart/) for computing sensitivity maps (external, not included)
- NVIDIA GPU + [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU acceleration (optional)

### Installation

```bash
git clone https://github.com/your-username/mslr-recon.git
cd mslr-recon
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Running a reconstruction

Copy an existing experiment file, update the paths and scan parameters, and run:

```bash
cp experiments/20251106_balltap.jl experiments/my_experiment.jl
# edit my_experiment.jl — set paths, N, Nvc, Nt, FOV, and use_gpu
```

**CPU** (multi-threaded patch SVDs):
```bash
julia -t auto experiments/my_experiment.jl
```

**GPU** (cuFFT for A/A' + CUSOLVER for patch SVDs, recommended):
```bash
julia experiments/my_experiment.jl
```

Set `use_gpu = true` or `false` inside the experiment file to choose the backend.
Output is saved as `<fn_recon_base>_<Nscales>scales.mat`.

### Analysing the result

```bash
julia scripts/analyze.jl /path/to/recon_3scales.mat
```

Optional flags:
```bash
--no-detrend        # skip linear drift removal before tSNR computation
--show-components   # display each scale component individually
```

---

## Writing an experiment file

Each experiment file calls `run_recon(; ...)` with keyword arguments. Here is a minimal example:

```julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Unitful: mm

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
using .Reconstruct

run_recon(
    fn_ksp          = "/data/my_experiment/kspace.mat",
    fn_smaps        = "/data/my_experiment/smaps_bart.mat",
    fn_recon_base   = "/data/my_experiment/recon",
    N               = (90, 90, 60),       # image matrix (Nx, Ny, Nz)
    Nvc             = 18,                 # virtual coils after BART compression
    Nt              = 300,                # number of time frames
    FOV             = (216mm, 216mm, 144mm),
    N_gre           = (108, 108, 108),    # GRE reference matrix (for smap cropping)
    FOV_gre         = (216mm, 216mm, 216mm),
    PATCH_SIZES     = [[90,90,60], [10,10,10]],   # one component per scale
    STRIDES         = [[90,90,60], [10,10,10]],   # non-overlapping
    NITERS          = 50,
    σ1A_PRECOMPUTED = 1.0,               # set to `nothing` to compute via power iteration
    use_gpu         = true,              # false for CPU
)
```

### Patch schedule guide

| Schedule | `PATCH_SIZES` | When to use |
|:---------|:-------------|:------------|
| Single-scale local LR | `[[10,10,10]]` | Fastest; good starting point |
| Global + local | `[[90,90,60], [10,10,10]]` | When global temporal drift is present |
| Full multi-scale | `[[90,90,60],[30,30,30],[10,10,10],[6,6,6],[1,1,1]]` | Maximum expressivity |

Set `STRIDES = PATCH_SIZES` for non-overlapping patches (fastest). Use
`STRIDES = [cld.(p, 2) for p in PATCH_SIZES]` for half-overlapping (smoother boundaries).

---

## GPU acceleration

GPU support requires [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl):

```bash
julia -e 'using Pkg; Pkg.add("CUDA")'
```

When `use_gpu = true`, the reconstruction:
- Moves sensitivity maps and k-space to GPU with `cu()`
- Replaces `MIRT.Asense` (FFTW-based) with `Asense_gpu` (cuFFT-based, defined in `src/sense_gpu.jl`)
- Runs patch SVDs sequentially via CUSOLVER instead of multi-threaded LAPACK
- Brings results back to CPU before saving

The GPU operator `Asense_gpu` uses the same FFT convention and normalisation as
`MIRT.Asense` with `fft_forward=true, unitary=true`, giving $\sigma_1(\mathcal{A}) \approx 1$.

**Memory requirements** for `N=(90,90,60)`, `Nt=300`, ComplexF32:

| Nscales | X size | Recommended GPU |
|:--------|:-------|:----------------|
| 1 | ~11 GB | RTX 3090 / A5000 |
| 3 | ~33 GB | RTX A6000 (48 GB) ✓ |
| 5 | ~55 GB | A100 80 GB |

---

## Input data format

| File | Format | Key | Shape |
|:-----|:-------|:----|:------|
| K-space | HDF5-backed `.mat` (v7.3) | `ksp_epi_zf` | `(Nx, Ny, Nz, Nvc, Nt)` ComplexF32 |
| Sensitivity maps | `.mat` | `smaps_raw` | `(Nx_gre, Ny_gre, Nz_gre, Nvc)` complex |

Zero entries in the k-space file are treated as unsampled. The sampling mask is
inferred automatically.

## Output file format

| Key | Shape | Description |
|:----|:------|:------------|
| `X_recon` | `(Nx, Ny, Nz, Nt)` | **Summed reconstruction — use this** |
| `X` | `(Nx, Ny, Nz, Nt, Nscales)` | Individual scale components |
| `omega` | `(Nx, Ny, Nz, Nt)` Bool | k-space sampling mask |
| `dc_costs` | `(Niters+1,)` | Data-consistency cost per iteration |
| `reg_costs` | `(Niters+1,)` | Regularisation cost per iteration |
| `restarts` | `(Niters+1,)` Bool | POGM restart events |
| `lambdas` | `(Nscales,)` | Per-scale regularisation weights |
| `scale_factor` | scalar | k-space normalisation constant |
| `sigma1A` | scalar | Spectral norm of $\mathcal{A}$ |
| `R` | scalar | Acceleration factor |
| `used_gpu` | Bool | Whether GPU was used |

---

## Tips

**λ is automatic.** The Ong & Lustig formula calibrates thresholds from patch
geometry and Nt. It works correctly as long as k-space is normalised, which
the reconstruction does internally (using the 99th-percentile image intensity).

**Lipschitz constant.** `σ₁(A) ≈ 1.0` for a unitary FFT-based SENSE operator.
Set `σ1A_PRECOMPUTED = nothing` on the first run to compute it via power
iteration (~20 min), then hard-code the printed value for future runs on the
same acquisition geometry.

**Memory.** Reduce the number of scales or set `use_gpu = false` and reduce
`-t` threads if memory is tight.

**Convergence.** 50 iterations is typically sufficient. Watch the convergence
plot from `analyze.jl` — if the total cost is still dropping steeply at the
end, increase `NITERS`.

**REPL workflow.** Experiment files use `Revise.includet` so you can re-run
them in the same REPL session without restarting Julia. Revise also
automatically picks up edits to `src/` files while the REPL is open.

---

## References

Ong, F. & Lustig, M. (2016). Beyond low rank + sparse: Multiscale low rank
matrix decomposition. *IEEE Journal of Selected Topics in Signal Processing*,
10(4), 672–687. https://doi.org/10.1109/JSTSP.2016.2545518

Kim, D. & Fessler, J.A. (2018). Adaptive restart of the optimized gradient
method for convex optimization. *Journal of Optimization Theory and
Applications*, 178(1), 240–263. https://doi.org/10.1007/s10957-018-1287-4

---

## Acknowledgements

The POGM implementation (`src/mirt_mod.jl`) is adapted from
[MIRT.jl](https://github.com/JeffFessler/MIRT.jl)
(Donghwan Kim & Jeff Fessler, University of Michigan).
