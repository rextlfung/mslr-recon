# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Run a reconstruction (CPU, multi-threaded):**
```bash
julia -t auto experiments/<experiment>.jl
```

**Run a reconstruction (GPU, recommended):**
```bash
julia experiments/<experiment>.jl
```

**Analyse a completed reconstruction:**
```bash
julia scripts/analyze.jl /path/to/recon_3scales.mat
julia scripts/analyze.jl /path/to/recon_3scales.mat --no-detrend --show-components
```

**REPL workflow with hot-reload (experiment files use `Revise.includet`):**
Edits to `src/` files are picked up automatically while the REPL is open. Re-include the experiment file to re-run without restarting Julia.

## Architecture

### Entry point and module loading

`scripts/reconstruct.jl` is the reconstruction module. It defines `module Reconstruct` and exports `run_recon(; ...)`. It does **not** have a `main()` — experiment files in `experiments/` call `run_recon(...)` directly after `Revise.includet`-ing the script.

At module load time, `reconstruct.jl` conditionally loads GPU support:
```julia
const _CUDA_AVAILABLE = try
    using CUDA
    include("src/sense_gpu.jl")
    CUDA.functional()
catch; false end
```
If CUDA.jl is not installed or no GPU is present, `use_gpu=true` raises a clear error at runtime.

### CPU vs GPU dispatch

The codebase uses Julia's type dispatch throughout — no `if use_gpu` branches inside hot loops:

- **`src/recon.jl`**: `_svst_loop!(P::Array, ...)` uses `@threads`; `_svst_loop!(P::AbstractArray, ...)` runs sequentially (CuArrays can't use `@threads`). `similar(img, ...)` ensures patch allocations stay on the same device as the input. `_svst_reconstruct` dispatches on `::Array` (uses `findall`, CPU-efficient) vs `::AbstractArray` (full matrix multiply, GPU-safe).
- **`src/sense_gpu.jl`**: Contains only `Asense_gpu` — the cuFFT-based SENSE operator. All patch/SVST operations use the `src/recon.jl` implementations directly, which already dispatch correctly for CuArrays.
- **`scripts/reconstruct.jl`**: `patchSVST` is called unconditionally; it dispatches to the right backend via the array type of `X[:,:,:,:,k]`.

The SENSE operator is the only thing that differs by backend: `Asense_gpu` (cuFFT) vs `MIRT.Asense` (FFTW). Both are wrapped in `LinearMapsAA.block_diag` to form a block-diagonal operator over time frames.

The sampling mask `Ω` is moved to GPU (`cu(Ω)`) alongside k-space so that k-space indexing (`ksp_flat[vec(s), :, it]`) stays device-resident and avoids CUDA.jl scalar indexing.

### Data flow inside `run_recon`

1. Load and normalize sensitivity maps (L2 norm per voxel across coils).
2. Load k-space from HDF5-backed `.mat` (key `ksp_epi_zf`), cast to `ComplexF32`.
3. Normalize k-space by 99th-percentile image intensity (via zero-filled SENSE combination).
4. Infer sampling mask `Ω` from zero entries; validate mask consistency across coils and frames.
5. Optionally move data to GPU.
6. Build block-diagonal SENSE operator `A` (one block per time frame).
7. Flatten and mask k-space to `(K, Nc, Nt)`.
8. Compute or reuse Lipschitz constant `L = Nscales × σ₁(A)²`.
9. Compute per-scale regularization weights `λ_k` via the Ong & Lustig formula.
10. Run POGM (`pogm_mod` from `src/mirt_mod.jl`) with the patch-SVST proximal operator applied independently to each scale component; progress is shown via `@showprogress` inside `pogm_mod`.
11. Save output as `<fn_recon_base>_<Nscales>scales.mat`.

The reconstruction variable `X` has shape `(Nx, Ny, Nz, Nt, Nscales)`. The cost function operates on `image_sum(X) = sum(X; dims=5)` for data consistency, but the proximal step acts on each scale independently.

### Optimizer — `pogm_mod` + `poweriter_mod` (`src/mirt_mod.jl`)

The reconstruction uses `pogm_mod` and `poweriter_mod` from `src/mirt_mod.jl` — GPU-compatible ports of `MIRT.pogm_restart` and `MIRT.poweriter`. `MIRT.pogm_restart` cannot run on GPU because it allocates gradients with `zeros(size(x0))` (creates a CPU Float64 array), uses Float64 scalar literals that would promote `CuArray{ComplexF32}` to `ComplexF64`, and uses `real(-Fgrad .* ynew_yold)` which allocates large intermediate CuArrays. `mirt_mod.jl` fixes all three, plus uses fused broadcast (`@.`) for the `znew` momentum update and related expressions to eliminate peak intermediate allocations of ~12–17 GB per iteration. Progress display is driven by `@showprogress` inside `pogm_mod`. The `fun` return values `(dc_cost, reg_cost, is_restart)` are collected into the `costs` array and unpacked for saving/plotting.

### `src/analysis.jl`

Provides `tSNR`, `detrend!`, and `plotOpt`. `detrend!` removes linear temporal drift per voxel in-place before tSNR computation.

## Key implementation details

**`σ₁(A) ≈ 1.0`** for a unitary FFT-based SENSE operator. Hard-code `σ1A_PRECOMPUTED = 1.0` after the first run to skip power iteration (~20 min).

**Patch boundary handling**: `img2patches` uses `cld` (ceiling division) for step counts and clamps the last patch origin to `Nx - psx + 1`, so the image is always fully covered even when dimensions are not multiples of the stride.

**Half-overlapping patches**: Set `STRIDES = [cld.(p, 2) for p in PATCH_SIZES]`. This improves boundary smoothness at the cost of more patches and longer runtimes.

**Peak memory formula (GPU and CPU)**

The dominant terms are the 9 simultaneously-live POGM iteration buffers and the `dc_cost` transients:

```
peak = N_opt × |X| + |img| + 3×|ksp| + persistent
```

| Symbol | Definition | Source |
|--------|-----------|--------|
| `\|X\|` | `Nx·Ny·Nz·Nt·Nscales × 8 B` | Full 5-D reconstruction tensor; POGM holds 9 simultaneous copies (xold≡zold, yold, uold, unew, xnew≡znew, ynew, Fgrad, Fgradold, ynew_yold) |
| `N_opt` | 9 (POGM), 6 (FISTA), 5 (ISTA) | All with gradient restart |
| `\|img\|` | `\|X\| / Nscales` | `image_sum(X)` transient inside `dc_cost` |
| `\|ksp\|` | `K·Nvc·Nt × 8 B`, K = Nx·Ny·Nz/R | `Ax` and `Ax−ksp` briefly coexist during `dc_cost` (×2 transients); plus the stored k-space array (×1 persistent) = ×3 total |
| `persistent` | smaps + ksp + Ω + idx | smaps: `Nx·Ny·Nz·Nvc×8 B`; Ω: `Nx·Ny·Nz·Nt×1 B`; idx (GPU only): `K·Nt×4 B` |

**Example — 20260409tap** (N=90×90×60, Nt=387, Nvc=21, R≈6, Nscales=2):
- `|X|` = 3.01 GB, `|img|` = 1.51 GB, `|ksp|` = 5.27 GB, persistent = 5.66 GB
- peak = 9×3.01 + 1.51 + 3×5.27 + 0.40 = **44.8 GB** (observed ~44 GB ✓)

The peak occurs in `mirt_mod.jl` at line 160 (`Fcostnew = Fcost(xnew)`) — `ynew` has just been allocated (line 159) so 9 POGM buffers are live, then `dc_cost` allocates `image_sum + Ax + residual`. Note `|ksp|` is independent of `Nscales`; adding scales raises the POGM buffer term but not the k-space transients.

**CPU vs GPU** (the formula applies to both VRAM and RAM):
1. **Forced GC**: `use_gpu && (GC.gc(true); CUDA.reclaim())` fires before each `g_prox` on GPU (reconstruct.jl:212), freeing `dc_cost_grad` temporaries. CPU lacks this; in the worst case, `|img| + 2|ksp|` ≈ 12 GB of temps may linger into `g_prox`, though Julia's allocation-triggered GC usually reclaims them before they accumulate.
2. **Unit-patch patchSVST** (`[1,1,1]`): GPU dispatches to the vectorized broadcast path (negligible allocation); CPU dispatches to `img2patches`, allocating `|img|` for the full patch tensor during `g_prox`.
3. **reg_cost logging**: CPU evaluates the full nuclear-norm cost each iteration (patch tensors ≈ `|img|` per scale at logger time); GPU returns 0. Does not raise the peak above the Fcost level.
4. **idx arrays**: 0.125 GB GPU-only overhead (`Asense_gpu` pre-computes Int32 gather indices; `MIRT.Asense` uses the Bool mask directly).
5. **Resource type**: VRAM is a hard limit (24–80 GB); RAM is typically 64–512 GB with swap as overflow.

**Sensitivity map format**: `run_recon` reads the key `"smaps"` (not `"smaps_raw"`) from `fn_smaps`. The file is a `.mat` written by BART after compression to `Nvc` virtual coils.
