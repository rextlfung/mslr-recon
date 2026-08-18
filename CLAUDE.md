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

**Generate a reconstruction report:**
```bash
julia scripts/report.jl /path/to/recon.mat
julia scripts/report.jl /path/to/recon.mat --no-components
```
Writes `<prefix>_report.png` (2×2: convergence, rel_change, mean magnitude, tSNR), `<prefix>_report.txt` (parameters + convergence + image-quality summary), and `<prefix>_scale<k>.png` per scale into the same directory as the input `.mat`.

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
2. Load k-space (key `ksp_epi_zf`), cast to `ComplexF32`. Accepts either a `.mat` (MATLAB v7.3, HDF5-backed) or a `.h5` (e.g. sigpy) file — see `_load_array` below.
3. Infer sampling mask `Ω` from zero entries; validate mask consistency across coils and frames.
4. Optionally move data to GPU.
5. Build block-diagonal SENSE operator `A` (one block per time frame).
6. Flatten and mask k-space to `(K, Nc, Nt)`.
7. Compute or reuse Lipschitz constant `L = Nscales × σ₁(A)²`.
8. Compute per-scale regularization weights `λ_k` via the Ong & Lustig formula, then scale all weights by `λ_GLOBAL` (default 1.0). Input k-space is assumed to be prewhitened by BART (σ_ksp ≈ 1); since A is approximately unitary, σ_image ≈ 1 and the formula applies directly with no correction. The log term uses natural log here (Ong's reference code uses `log2`; ~2–3% effect on `λ_k`, absorbed by `λ_GLOBAL`). See [Math correctness](#math-correctness) below for the full audit.
9. Initialize `X0` by setting every scale slice to `(A' * ksp) / Nscales`, so the sum across scales equals the adjoint reconstruction.
10. Run `pogm_restart` (from `src/mirt_mod.jl`) with the momentum scheme selected by the `mom` parameter (default `:fpgm`; `:pogm` and `:pgm` also supported) and the patch-SVST proximal operator applied independently to each scale component; if `cycle_spin=true` (default `false`), each scale's volume is randomly `circshift`-ed in the spatial dimensions before `patchSVST` and exactly unshifted afterward — boundary artifacts average out over iterations (Figueiredo & Nowak 2003); unit patches `[1,1,1]` are excluded (shift is a no-op there). Progress is shown via `@showprogress` inside `pogm_restart`. The relative iterate change `‖x_new − x_prev‖_F / ‖x_prev‖_F` is computed every iteration (reusing already-live `yold`/`xold`) and logged via the `fun` callback. Early stopping fires when it falls below `conv_tol` (default `1e-5`) for the first time after `adaptive_cmi` iterations have elapsed since the last gradient restart, where `adaptive_cmi` starts at `conv_min_iter` (default 10) and decrements by 1 each time a restart fires (floored at 1). The decay prevents a chain of restarts near convergence from permanently blocking early stopping, while the initial value preserves the full grace period during genuine optimization progress. The iterate compared is the prox-step output (ynew for `:fpgm`/`:pgm`, xnew for `:pogm`); set `conv_tol=0` to disable (with `cycle_spin=true`, random shifts inflate `rel_change`, potentially preventing early stopping). Default `NITERS=200`. The whole `pogm_restart` call is wrapped in `time()` to capture wall-clock runtime.
11. Save output to `fn_recon` (the caller-specified full path). Persisted keys include the recon (`X`, `X_recon`, `omega`), per-iteration traces (`dc_costs`, `reg_costs`, `restarts`, `rel_changes`), parameters (`R`, `sigma1A`, `L`, `Nscales`, `patch_sizes`, `strides`, `lambdas`, `lambda_global`, `Niters`, `mom`, `conv_tol`, `cycle_spin`), and runtime metadata (`used_gpu`, `device`, `runtime_s`, `iter_time_s`). See README.md for the full key reference.

The reconstruction variable `X` has shape `(Nx, Ny, Nz, Nt, Nscales)`. The cost function operates on `image_sum(X) = sum(X; dims=5)` for data consistency, but the proximal step acts on each scale independently.

### Optimizer — `pogm_restart` + `poweriter` (`src/mirt_mod.jl`)

The reconstruction uses `pogm_restart` from `src/mirt_mod.jl`. The momentum scheme is selected by `run_recon`'s `mom` parameter (`:fpgm` default, `:pogm`, or `:pgm`). The prox step size is fixed at `α = 1/L` every iteration for `:fpgm`/`:pgm`, giving a fixed SVST threshold of `α × λ_k = λ_k / L`; `:pogm` uses a per-iteration `zetanew`. `poweriter` estimates the Lipschitz constant if not precomputed.

`pogm_restart` is a modified port of `MIRT.pogm_restart`. `MIRT.pogm_restart` cannot run on GPU because it allocates gradients with `zeros(size(x0))` (creates a CPU Float64 array), uses Float64 scalar literals that would promote `CuArray{ComplexF32}` to `ComplexF64`, and uses `real(-Fgrad .* ynew_yold)` which allocates large intermediate CuArrays. `mirt_mod.jl` fixes all three, and additionally adds early stopping via `conv_tol`. Progress display is driven by `@showprogress` inside `pogm_restart`. The `fun` callback is invoked with `(iter, xk, yk, is_restart, Fcostnew, rel_change)` — one more positional arg than upstream MIRT, where `rel_change` is the per-iteration `‖Δx‖/‖x‖` (NaN at iter 0). `reconstruct.jl`'s logger returns `(Fcostnew, last_reg[], is_restart, rel_change)`, which are collected into the `costs` array and unpacked into `dc_costs`, `reg_costs`, `restarts`, `rel_changes` for saving/plotting.

### `scripts/report.jl` and `src/metrics.jl`

`src/metrics.jl` provides reusable utilities (`tSNR`, `plotOpt`). `scripts/report.jl` defines `run_report(fn_recon; show_components=true)`, which loads a recon `.mat` and writes three artifacts to the same directory as the input `.mat`, with a filename prefix derived from its basename:
- `<prefix>_report.png` — single 2×2 figure: convergence, relative iterate change (log-y, with `conv_tol` reference line), mean magnitude montage, tSNR montage.
- `<prefix>_report.txt` — parameters + convergence + image-quality stats.
- `<prefix>_scale<k>.png` — per-scale mean magnitude (omitted when `show_components=false` or `Nscales == 1`).

Older `.mat` files lacking the newer metadata keys (`rel_changes`, `runtime_s`, `device`, `mom`, `conv_tol`, `cycle_spin`) are handled via `haskey` fallbacks — the rel-change panel renders an "n/a" placeholder and missing summary lines are omitted.

## Key implementation details

**`σ₁(A) ≤ 1.0`** always (subsampling can only reduce the norm of the unsubsampled unitary operator). The unsubsampled operator is exactly unitary (σ₁ = 1), but subsampling with an incoherent mask reduces it slightly. Empirically, `σ₁(A) ≈ 0.968` for the 20260409tap dataset (measured via `tests/sigma1A_tests.jl`). Hard-code `σ1A = 0.968294` after the first run to skip power iteration (~20 min). Using 1.0 is safe (overestimates L, so step size is conservative) but ~6.7% suboptimal.

**Patch boundary handling**: `img2patches` uses `cld` (ceiling division) for step counts and clamps the last patch origin to `Nx - psx + 1`, so the image is always fully covered even when dimensions are not multiples of the stride.

**Half-overlapping patches**: Set `STRIDES = [cld.(p, 2) for p in PATCH_SIZES]`. This improves boundary smoothness at the cost of more patches and longer runtimes.

**Peak memory formula (GPU and CPU)**

The dominant terms are the simultaneously-live FISTA iteration buffers and the `dc_cost` transients:

```
peak = N_opt × |X| + |img| + 3×|ksp| + persistent
```

| Symbol | Definition | Source |
|--------|-----------|--------|
| `\|X\|` | `Nx·Ny·Nz·Nt·Nscales × 8 B` | Full 5-D reconstruction tensor; FPGM holds 6 simultaneous copies during Fcost (x0, xold, yold, fgrad, Fgrad, ynew); POGM additionally allocates Fgradold, ynew_yold, unew |
| `N_opt` | 6 (FPGM), 9 (POGM), 5 (PGM) | During Fcost: x0 (held by caller's X0 binding), xold, yold, fgrad, Fgrad, ynew; Fgradold/ynew_yold alias x0 for FPGM/PGM (zero extra cost) |
| `\|img\|` | `\|X\| / Nscales` | `image_sum(X)` transient inside `dc_cost` |
| `\|ksp\|` | `K·Nvc·Nt × 8 B`, K = Nx·Ny·Nz/R | `Ax` and `Ax−ksp` briefly coexist during `dc_cost` (×2 transients); plus the stored k-space array (×1 persistent) = ×3 total |
| `persistent` | smaps + Ω + idx | smaps: `Nx·Ny·Nz·Nvc×8 B`; Ω: `Nx·Ny·Nz·Nt×1 B`; idx (GPU only): `K·Nt×4 B` (ksp's own persistent copy is already counted in the ×3 of `\|ksp\|` above) |

**Example — 20260409tap** (N=90×90×60, Nt=375, Nvc=21, R≈6, Nscales=1):
- `|X|` = 1.46 GB, `|img|` = 1.46 GB, `|ksp|` = 5.10 GB, persistent = 0.39 GB
- peak = 6×1.46 + 1.46 + 3×5.10 + 0.39 = **25.9 GB** (vs ~30.3 GB with POGM)

The peak occurs during `Fcostnew = Fcost(ynew)` — 6 FPGM buffers are live (x0, xold, yold, fgrad, Fgrad, ynew) and `dc_cost` allocates `image_sum + Ax + residual`. Note `|ksp|` is independent of `Nscales`; adding scales raises the optimizer buffer term but not the k-space transients.

**CPU vs GPU** (the formula applies to both VRAM and RAM):
1. **Forced GC**: `use_gpu && (GC.gc(true); CUDA.reclaim())` fires before each `g_prox` on GPU (reconstruct.jl:249), freeing `dc_cost_grad` temporaries. CPU lacks this; in the worst case, `|img| + 2|ksp|` ≈ 12 GB of temps may linger into `g_prox`, though Julia's allocation-triggered GC usually reclaims them before they accumulate.
2. **Unit-patch patchSVST** (`[1,1,1]`): both CPU and GPU use the shared `_unit_block_svst` vectorized broadcast (negligible allocation) — no backend difference here. The full `|img|` patch-tensor allocation happens only for **non-unit** patches on CPU (`img2patches` builds the whole tensor for `@threads`), whereas GPU streams one patch at a time.
3. **reg_cost logging**: Both CPU and GPU evaluate the full nuclear-norm cost each iteration. On GPU, `xk` is copied to CPU before calling `reg_cost` (patch tensors ≈ `|img|` per scale) to keep the large intermediate patch tensor off-device. Does not raise the peak VRAM above the Fcost level.
4. **idx arrays**: 0.125 GB GPU-only overhead (`Asense_gpu` pre-computes Int32 gather indices; `MIRT.Asense` uses the Bool mask directly).
5. **Resource type**: VRAM is a hard limit (24–80 GB); RAM is typically 64–512 GB with swap as overflow.

**Sensitivity map format**: `run_recon` reads the key `"smaps"` (not `"smaps_raw"`) from `fn_smaps`. The file is either a `.mat` written by BART or a `.h5` written by sigpy, both after compression to `Nvc` virtual coils.

**Input file formats (`.mat` / `.h5`)**: `fn_ksp` and `fn_smaps` each accept a `.mat` (MATLAB v7.3, HDF5-backed — BART's output) or a `.h5` (written by a row-major tool such as Python/h5py — e.g. sigpy's output) file, dispatched by extension in `_load_array` (`scripts/reconstruct.jl`). Both formats are read via `h5read`, but need different correction: MATLAB's HDF5 writer stores array dims C-order-reversed on disk, which `HDF5.jl` reverses back on read, restoring MATLAB's logical (column-major) dimension order automatically — and MATLAB stores complex numbers as a `{real, imag}` struct, reassembled with `complex.(v.real, v.imag)`. Row-major writers store dims in the same order as `HDF5.jl` reports them on disk (no pre-reversal), so `HDF5.jl`'s automatic reversal on read must itself be undone with `permutedims(raw, reverse(1:ndims(raw)))` to recover the logical (numpy) shape; complex numbers are stored natively, no struct reassembly needed. Verified on `20260810ball`'s paired BART `.mat` / sigpy `.h5` exports: bit-identical sampling masks after correction.

**Experiment file structure**: All tunable parameters (`PATCH_SIZES`, `STRIDES`, `NITERS`, `σ1A`, `MOMENTUM`, `TOL`, `CYCLE_SPIN`) are declared as `const` at the top of each experiment file. Newer experiments also declare a `LAMBDA_SWEEP` array of `(λ_GLOBAL, subfolder)` tuples and loop over `LAMBDA_SWEEP × datasets`, with `try/catch` around each reconstruction for error resilience. Each experiment guards `run_recon` with a three-branch check: output missing → run; output exists and `params_match(fn_out; ...)` returns true → skip recon, regenerate report; output exists but params differ → `@warn` and skip (no overwrite). `params_match` (from `utils/recon_cache.jl`) checks `NITERS`, `PATCH_SIZES`, `STRIDES`, `σ1A`, `mom`, `conv_tol`, `lambda_global`, and `cycle_spin`. To run with new parameters, shelve the old output to a subdirectory first.

## Math correctness

Review of the multi-scale low-rank (MSLR) fMRI reconstruction for mathematical
correctness, cross-checked against Ong & Lustig (2016) and the authors' reference
implementation (`frankong/multi_scale_low_rank`).

**Bottom line: no mathematical errors were found in the core algorithm.** The
reconstruction solves the intended composite problem correctly. Findings are minor —
one log-base fidelity choice, one latent unused code path, and a missing (deliberately
deferred) anti-blocking technique. This review documents what was verified and to what
depth, so the claims are not over-broad.

The problem solved is

$$\min_{\mathbf X}\; \tfrac12\big\|\,\mathcal A\big(\textstyle\sum_k \mathbf X_k\big)-\mathbf y\,\big\|_2^2 \;+\; \sum_k \lambda_k\,\big\|\mathcal P_k(\mathbf X_k)\big\|_*$$

with one image component $\mathbf X_k$ per spatial scale, data consistency on the sum, and
a nuclear-norm penalty on the (voxels × time) patch matrices of each component.

---

### Verified correct — in depth

- **Composite objective & separability of the prox.** The scales are independent
  variables, so the proximal operator separates across scales. For **non-overlapping
  patches that tile the volume** (the default `STRIDES = PATCH_SIZES`, when each patch size
  divides its dimension — true for all provided experiments: 90/6, 60/6, full-volume, 1³) it
  further separates across patches, so `g_prox` (per-scale, per-patch SVST with
  overlap-averaging) is the **exact** proximal operator of the regularizer. With overlapping
  patches — or a non-dividing patch size, since `img2patches` clamps the last patch origin and
  thus overlaps at the boundary — it is the standard LLR overlap-averaging approximation.

- **Gradient.** `dc_cost_grad(X) = repeat(A'(A·ΣₖXₖ − y))` is exactly $S'\mathcal A'(\mathcal A\,S\,\mathbf X-\mathbf y)$,
  where $S:\mathbf X\mapsto\sum_k\mathbf X_k$ is the sum-over-scales operator and $S'$ its
  adjoint (replicate to all scales). ✓

- **Lipschitz constant.** $L = N_{\text{scales}}\,\sigma_1(\mathcal A)^2$ is the **tight**
  bound. $\|\mathcal A S\| = \sqrt{N_{\text{scales}}}\,\sigma_1(\mathcal A)$ (Cauchy–Schwarz,
  attained when all scale components equal A's top right singular vector), hence
  $\|S'\mathcal A'\mathcal A S\| = N_{\text{scales}}\,\sigma_1(\mathcal A)^2$. The prox step
  $\alpha = 1/L$ gives SVST threshold $\lambda_k/L$. ✓

- **FISTA / FPGM momentum** (`mom=:fpgm`, default). The gradient is evaluated at the
  momentum point, the prox produces the iterate, and the $t$-update and
  $\beta=(t-1)/t^{+}$ are standard. Variable naming is transposed relative to the textbook
  (here `x` is the momentum point and `y` the prox output) but the recursion is FISTA. The
  first iteration has $\beta=0$, and the routine returns the prox output (a genuinely
  low-rank iterate), not the extrapolated momentum point. ✓

- **Gradient restart** (`:gr`, default). $\mathrm{Fgrad}=(1/\alpha)(x_{\text{old}}-y_{\text{new}})$
  is the prox-gradient *mapping* (it accounts for both $f$ and $g$), so the gradient-restart
  test is valid even though `Fcost` tracks only the smooth data term. ✓

- **SVST** is singular-value soft-thresholding = the nuclear-norm proximal operator
  (Cai–Candès–Shen). ✓

- **Unit-patch `[1,1,1]` fast path** = block soft-threshold of each voxel's time series.
  The SVD of a $1\times N_t$ row is $U=[1],\,S=[\|x\|],\,V^\mathsf{H}=x/\|x\|$, so SVST gives
  $\max(1-\beta/\|x\|,0)\cdot x$ with nuclear norm $\|x\|_2$. The vectorized broadcast computes
  exactly this (and avoids ~$N_{\text{vox}}$ serial GPU SVDs). It is joint/group sparsity across
  time, not voxel-wise $\ell_1$. ✓

- **GPU SENSE adjoint scaling.** Forward scale $1/\sqrt N$; adjoint scale $\sqrt N$ (since the
  adjoint of the unnormalized `fft` is $N\cdot\text{ifft}$), with `fftshift`↔`ifftshift`
  correctly swapped. Confirmed numerically by the `⟨Ax,y⟩ ≈ ⟨x,A'y⟩` test in
  `tests/kernel_tests.jl`. ✓

- **λ formula structure matches the reference.** The reference code computes
  `√ms + √ns + √(log₂(bs·min(ms,ns)))` with `ms = p_k`, `ns = Nt`, `bs = N_vox/p_k`. Its log
  argument `bs·min(ms,ns) = N_vox·Nt/max(p_k,Nt)` is **algebraically identical** to this code's
  argument, because `min(a,b)/a = b/max(a,b)`. The `√p_k + √Nt` terms match exactly. ✓ (Only the
  log *base* differs — see finding 1.)

### Verified by spot-check (documented port, not re-derived term-by-term)

- **POGM** (`mom=:pogm`) and **`poweriter`** are faithful GPU/memory-efficient ports of
  `MIRT.pogm_restart` (Kim & Fessler 2018); the structure matches and the modifications are
  documented in the module header. They were not independently re-derived line-by-line.

---

### Minor findings

1. **Natural log vs base-2 (a documentation choice, not a bug).** Ong & Lustig's eq. (4) is
   base-agnostic (stated up to a constant, "~"), so it does not specify a log base. The
   authors' reference code uses `log2` (verified verbatim in `matlab/demo_dce_mri_decom.m`
   and `matlab/demo_hanning_decom.m`). `reconstruct.jl` uses natural `log`, which shrinks the
   third term by $\sqrt{\ln 2}\approx0.83$ (reference term ~1.20× larger) — a **~2–3% net
   change in $\lambda_k$**, since $\sqrt{N_t}$ dominates the log term (and $\sqrt{p_k}$
   dominates at the global scale). Natural log is consistent with the paper, and the empirical
   `λ_GLOBAL` absorbs global rescaling. Switching to `log2` is an optional fidelity tweak only.

2. **`restart=:fr` is latent-inconsistent.** Function restart compares `Fcost` (the smooth
   data term passed by `reconstruct.jl`) and so would restart on the data term alone, not the
   full $f+g$ objective. Harmless in practice: the default `:gr` is correct and `:fr` is never
   selected. A clarifying comment was added at the restart block.

3. **Random cycle spinning is implemented** (`cycle_spin=true` in `run_recon`; default `false`). See below.

4. **`σ1A = 1.0` vs the measured 0.968** overestimates $L$ by ~6.7%, giving a
   conservative (smaller) step. Safe and already documented. No change.

---

### Random cycle spinning

Each `g_prox` call draws an independent random spatial shift $(Δx, Δy, Δz)$ from $[0,N_x-1]\times[0,N_y-1]\times[0,N_z-1]$, applies `circshift` before `patchSVST`, then exactly inverts with `circshift` using the negated shift. Over many iterations the shifts are i.i.d., so patch-boundary artifacts average out in expectation (Figueiredo & Nowak 2003, *IEEE TIP* 12(8):906–916; Coifman & Donoho 1995). Unit patches `[1,1,1]` are excluded — SVST is separable per voxel there, making shift/unshift a provable no-op.

**Convergence caveat.** Cycle spinning makes the proximal map stochastic: the iterates converge to a noise ball around the minimizer of the shift-averaged objective, not a fixed point. Consequently `rel_change` has a positive noise floor and `conv_tol` early-stopping will likely never fire. Set `conv_tol=0` when using `cycle_spin=true`; `run_recon` issues a `@warn` if both are active simultaneously. Ong & Lustig / BART do run FISTA + cycle spinning successfully, so the cost trace should still descend — verify on an actual run.

**Reproducibility.** `cycle_spin=true` is non-deterministic (Julia's default RNG is not seeded by `run_recon`). Same parameters now give different outputs each run. `params_match` still caches correctly — it checks parameter equality, not output equality.

### Optional future tweak

**`log` → `log2`** in the `λ_k` formula (`reconstruct.jl` §7) for bit-level fidelity to the reference implementation. Shifts $\lambda_k$ by ~2–3% and changes future recon outputs; existing saved `.mat` files are unaffected.
