# mslr-recon

**Multi-scale Low-Rank (MSLR) Reconstruction in Julia**

Iterative reconstruction of 3D + time MRI data. Uses a SENSE forward model and a **multi-scale low-rank decomposition** regularizer optimized with a proximal gradient method (FPGM by default, POGM available). Supports both CPU (multi-threaded) and GPU (CUDA) execution.

---

## Background

Accelerated MRI acquisitions collect only a fraction of k-space at each time frame. Recovering a clean image from undersampled data requires exploiting the structure of the signal — in this case, the fact that nearby voxels tend to have correlated temporal dynamics as they belong to the same functional nuclei, which can be modelled as a form of local low-rank structure.

This codebase implements the **multiscale low-rank matrix decomposition** of [Ong & Lustig (2016)](#references), which represents the image as a sum of components each regularized at a different spatial scale:

```
X_final = X_global + X_regional + X_local + ...
```

Each component independently captures low-rank temporal structure at its own patch size, from a single global component spanning the whole volume down to small local patches and single voxels (sparsity). Data consistency is enforced on the sum. This is more expressive than promoting low-rankness at different scales sequentially, as the scales cannot interfere with each other.

---

## Algorithm

The reconstruction solves:

$$\min_{\mathbf{X}} \; \frac{1}{2} \left\| \mathcal{A}\!\left(\sum_k \mathbf{X}_k\right) - \mathbf{Y} \right\|_F^2 \;+\; \sum_k \lambda_k \left\| \mathcal{P}_k(\mathbf{X}_k) \right\|_*$$

where:
- $\mathcal{A}$ is the block-diagonal SENSE encoding operator (one block per time frame)
- $\mathbf{Y}$ is the measured k-space data shaped as a (space x time) matrix
- $\mathcal{P}_k(\mathbf{X}_k)$ extracts and reshapes spatial patches of component $k$ as (voxels × time) matrices
- $\|\cdot\|_*$ is the nuclear norm, which is the convex relaxation of the rank of a matrix.
- $\lambda_k$ is set automatically via the Ong & Lustig (2016) formula — no manual tuning needed:

$$\lambda_k = \sqrt{p_k} + \sqrt{N_t} + \sqrt{\log\!\left(\frac{N_{vox} \cdot N_t}{\max(p_k,\, N_t)}\right)}$$

where $p_k$ is the number of voxels in a patch at scale $k$.

Optimization uses `pogm_restart` (from `src/mirt_mod.jl`) with gradient restart. The momentum scheme is configurable via the `mom` parameter (`:fpgm` default, `:pogm` for the Proximal Optimized Gradient Method, `:pgm` for plain gradient descent). The Lipschitz constant is $L = N_{scales} \cdot \sigma_1(\mathcal{A})^2$.

---

## Repository structure

```
mslr-recon/
├── Project.toml              # Julia package dependencies
│
├── src/
│   ├── recon.jl              # Patch extraction/recombination, SVST, k-space utilities
│   ├── analysis.jl           # tSNR maps, convergence plots
│   └── sense_gpu.jl          # GPU-native SENSE operator (requires CUDA.jl)
│
├── scripts/
│   ├── reconstruct.jl        # Reconstruction module — called by experiment files
│   └── analyze.jl            # Post-reconstruction analysis and visualization
│
└── experiments/
    ├── 20241017tap.jl        # Finger-tapping, 10 coils, Nt=300
    ├── 20251106balltap.jl    # Ball phantom + finger-tapping, 18 coils, Nt=300
    ├── 20260317tap.jl        # Finger-tapping, 18 coils, Nt=387, half-overlapping patches
    └── 20260409tap.jl        # Finger-tapping, 21 coils, Nt=387, 3-scale, half-overlapping; loops over 3 datasets (caipi, caipi_ts, pd), prints each before reconstructing
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
cp experiments/20251106balltap.jl experiments/my_experiment.jl
# edit my_experiment.jl — set paths and use_gpu
```

**CPU** (multi-threaded patch SVDs):
```bash
julia -t auto experiments/my_experiment.jl
```

**GPU** (cuFFT for A/A' + CUSOLVER for patch SVDs, recommended):
```bash
julia experiments/my_experiment.jl
```

Set `use_gpu = true` or `false` inside the experiment file to choose the backend. Output is saved as `<fn_recon_base>_<Nscales>scales.mat`.

### Analysing the result

```bash
julia scripts/analyze.jl /path/to/recon_3scales.mat
```

Optional flags:
```bash
--no-components     # skip per-scale component images
```

All plots are saved as PNGs to `plots/` (created automatically) with a filename prefix matching the input `.mat` basename (e.g. `plots/caipi_recon_2scales_tsnr.png`).

---

## Writing an experiment file

Each experiment file calls `run_recon(; ...)` with keyword arguments. Here is a minimal example:

```julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
using .Reconstruct

run_recon(
    fn_ksp          = "/data/my_experiment/kspace.mat",
    fn_smaps        = "/data/my_experiment/smaps_bart.mat",
    fn_recon_base   = "/data/my_experiment/recon",
    PATCH_SIZES     = [[90,90,60], [10,10,10]],   # one component per scale
    STRIDES         = [[90,90,60], [10,10,10]],   # non-overlapping
    σ1A_PRECOMPUTED = 1.0,               # set to `nothing` to compute via power iteration
    use_gpu         = true,              # false for CPU
    mom             = :fpgm,            # :fpgm (default), :pogm, or :pgm
    # NITERS and conv_tol have defaults (200 and 1e-5); override if needed:
    # NITERS          = 200,
    # conv_tol        = 1e-5,
)
```

Image dimensions (`Nx`, `Ny`, `Nz`), number of coils (`Nvc`), and number of frames (`Nt`) are inferred automatically from the k-space file. The sensitivity maps must match the k-space spatial dimensions and coil count — an assertion fires at load time if they don't.

### Patch schedule guide

| Schedule | `PATCH_SIZES` | When to use |
|:---------|:-------------|:------------|
| Single-scale local LR | `[[6,6,6]]` | Fastest; good starting point |
| Global + local | `[[90,90,60], [6,6,6], [1,1,1]]` | When global temporal drift is present |
| Full multi-scale | `[[90,90,60],[30,30,30],[10,10,10],[6,6,6],[1,1,1]]` | Maximum expressivity |

Set `STRIDES = PATCH_SIZES` for non-overlapping patches (fastest). Use `STRIDES = [cld.(p, 2) for p in PATCH_SIZES]` for half-overlapping (smoother boundaries).

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

The GPU operator `Asense_gpu` uses the same FFT convention and normalization as `MIRT.Asense` with `fft_forward=true, unitary=true`, giving $\sigma_1(\mathcal{A}) \leq 1$ (exactly 1 for the fully-sampled operator; slightly less under subsampling).

**Memory requirements**

Peak VRAM is set by three terms that are simultaneously live:

```
peak ≈ N_opt × |X| + |img| + 3 × |ksp| + persistent
```

where `|X| = Nx·Ny·Nz·Nt·Nscales × 8 B` (the full reconstruction tensor), `|img| = |X|/Nscales` (gradient transient), `|ksp| = (Nx·Ny·Nz/R)·Nvc·Nt × 8 B` (k-space appears 3×: 1 stored + 2 `dc_cost` transients), and `persistent` covers smaps, the sampling mask, and small index arrays. `N_opt` depends on `mom`: 6 for `:fpgm` (default), 9 for `:pogm`, 5 for `:pgm`.

**Worked example** — N=90×90×60, Nt=387, Nvc=21, R=6, Nscales=2, `mom=:fpgm` (default):

| Term | Size |
|:-----|-----:|
| 6 × \|X\| (FPGM buffers) | 18.1 GB |
| \|img\| (gradient transient) | 1.5 GB |
| 3 × \|ksp\| (k-space terms) | 15.8 GB |
| persistent (smaps + Ω + idx) | 0.5 GB |
| **Total** | **~35.9 GB** |

With `mom=:pogm` the buffer term rises to 9 × \|X\| ≈ 27.1 GB, giving ~44.9 GB total.

Note that `|ksp|` does not scale with `Nscales` — adding more scales raises only the POGM buffer term. For high Nvc or long Nt, the k-space terms can dominate. The same formula applies to CPU RAM (see `CLAUDE.md` for minor CPU/GPU differences); RAM is rarely the binding constraint.

---

## Input data format

| File | Format | Key | Shape |
|:-----|:-------|:----|:------|
| K-space | HDF5-backed `.mat` (v7.3) | `ksp_epi_zf` | `(Nx, Ny, Nz, Nvc, Nt)` ComplexF32 |
| Sensitivity maps | `.mat` | `smaps` | `(Nx, Ny, Nz, Nvc)` ComplexF32 |

Zero entries in the k-space file are treated as unsampled. The sampling mask is inferred automatically.

## Output file format

| Key | Shape | Description |
|:----|:------|:------------|
| `X_recon` | `(Nx, Ny, Nz, Nt)` | Reconstructed image as sum of all components |
| `X` | `(Nx, Ny, Nz, Nt, Nscales)` | Individual scale components |
| `omega` | `(Nx, Ny, Nz, Nt)` Bool | k-space sampling mask |
| `dc_costs` | `(Niters+1,)` | Data-consistency cost per iteration |
| `reg_costs` | `(Niters+1,)` | Regularization cost per iteration |
| `restarts` | `(Niters+1,)` | POGM restart events |
| `lambdas` | `(Nscales,)` | Per-scale regularization weights |
| `scale_factor` | scalar | k-space normalization constant |
| `sigma1A` | scalar | Spectral norm of $\mathcal{A}$ |
| `R` | scalar | Acceleration factor |
| `used_gpu` | Bool | Whether GPU was used |

---

## Tips

**λ is automatic.** The Ong & Lustig formula calibrates thresholds from patch geometry and Nt. It works correctly as long as k-space is normalized, which the reconstruction does internally (using the 99th-percentile image intensity).

**Lipschitz constant.** `σ₁(A) ≤ 1.0` always — the unsubsampled operator is exactly unitary but subsampling reduces the spectral norm slightly (empirically `σ₁(A) ≈ 0.968` for the 20260409tap dataset). Set `σ1A_PRECOMPUTED = nothing` on the first run to measure it via power iteration (~20 min via `tests/sigma1A_test.jl`), then hard-code the result. Using `1.0` is safe (conservative step size) but ~6.7% suboptimal.

**Memory.** If VRAM is tight, reduce `Nscales` (each scale adds `N_opt × Nx·Ny·Nz·Nt × 8 B` to the optimizer buffer, where `N_opt` is 6 for `:fpgm`, 9 for `:pogm`) or reduce `Nvc` at the BART sensitivity-map compression step. The k-space term `3 × |ksp|` is fixed regardless of `Nscales` or `mom`. Switch to `use_gpu = false` to use RAM instead of VRAM; set `-t auto` to use all CPU threads for the patch SVDs.

**Convergence.** Early stopping fires when the relative image-iterate change `‖x_new − x_prev‖_F / ‖x_prev‖_F` falls below `conv_tol` (default `1e-5`), with a 10-iteration warmup. The iterate compared is the prox-step output (before momentum extrapolation), which is more stable than the momentum point. The default `NITERS=200` acts as a hard cap. Set `conv_tol=0` to always run all iterations. For fMRI, underfitting (stopping too early) is the main risk — aliasing artifacts left by an underconverged reconstruction appear as spurious spatial structure that tSNR cannot detect, since temporally static artifacts do not inflate temporal variance.

**REPL workflow.** Experiment files use `Revise.includet` so you can re-run them in the same REPL session without restarting Julia. Revise also automatically picks up edits to `src/` files while the REPL is open.

---

## References

Full BibTeX entries are in [`REFERENCES.bib`](REFERENCES.bib). Key citations:

**Core algorithm**
- Ong & Lustig (2016). Beyond low rank + sparse: Multiscale low rank matrix decomposition. *IEEE J. Sel. Top. Signal Process.*, 10(4), 672–687. https://doi.org/10.1109/JSTSP.2016.2545518 — MSLR regularizer and λ_k formula.

**Optimizer**
- Kim & Fessler (2018). Adaptive restart of the optimized gradient method for convex optimization. *J. Optim. Theory Appl.*, 178(1), 240–263. https://doi.org/10.1007/s10957-018-1287-4 — POGM with gradient restart; ported in `src/mirt_mod.jl`.
- Beck & Teboulle (2009). A fast iterative shrinkage-thresholding algorithm. *SIAM J. Imaging Sci.*, 2(1), 183–202. https://doi.org/10.1137/080716542 — FISTA / FPGM (default `mom=:fpgm`).

**Forward model**
- Pruessmann et al. (1999). SENSE: sensitivity encoding for fast MRI. *Magn. Reson. Med.*, 42(5), 952–962. https://doi.org/10.1002/(SICI)1522-2594(199911)42:5<952::AID-MRM16>3.0.CO;2-S
- Fessler & Sutton (2003). Nonuniform fast Fourier transforms using min-max interpolation. *IEEE Trans. Signal Process.*, 51(2), 560–574. https://doi.org/10.1109/TSP.2002.807005 — NUFFT underlying `MIRT.Asense`.

**Related fMRI reconstruction**
- Chiew et al. (2015). Recovering task fMRI signals from highly under-sampled data with low-rank and temporal subspace constraints. *NeuroImage*, 114, 98–111. https://doi.org/10.1016/j.neuroimage.2015.03.055 — k-t FASTER; closest comparable fMRI reconstruction approach.

---

## Acknowledgements

The POGM optimizer uses `pogm_restart` from [MIRT.jl](https://github.com/JeffFessler/MIRT.jl) (Donghwan Kim & Jeff Fessler, University of Michigan).