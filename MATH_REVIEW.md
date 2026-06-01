# Mathematical Correctness Review — mslr-recon

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

## Verified correct — in depth

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

## Verified by spot-check (documented port, not re-derived term-by-term)

- **POGM** (`mom=:pogm`) and **`poweriter`** are faithful GPU/memory-efficient ports of
  `MIRT.pogm_restart` (Kim & Fessler 2018); the structure matches and the modifications are
  documented in the module header. They were not independently re-derived line-by-line.

---

## Minor findings

1. **Natural log vs base-2 (a documentation choice, not a bug).** Ong & Lustig's eq. (4) is
   base-agnostic (stated up to a constant, "~"), so it does not specify a log base. The
   authors' reference code uses `log2` (verified verbatim in `matlab/demo_dce_mri_decom.m`
   and `matlab/demo_hanning_decom.m`). `reconstruct.jl` uses natural `log`, which shrinks the
   third term by $\sqrt{\ln 2}\approx0.83$ (reference term ~1.20× larger) — a **~2–3% net
   change in $\lambda_k$**, since $\sqrt{N_t}$ dominates the log term (and $\sqrt{p_k}$
   dominates at the global scale). Natural log is consistent with the paper, and the empirical
   `λ_SCALE` absorbs global rescaling. Switching to `log2` is an optional fidelity tweak only.

2. **`restart=:fr` is latent-inconsistent.** Function restart compares `Fcost` (the smooth
   data term passed by `reconstruct.jl`) and so would restart on the data term alone, not the
   full $f+g$ objective. Harmless in practice: the default `:gr` is correct and `:fr` is never
   selected. A clarifying comment was added at the restart block.

3. **Random cycle spinning is not implemented (deferred — see Future work).**

4. **`σ1A_PRECOMPUTED = 1.0` vs the measured 0.968** overestimates $L$ by ~6.7%, giving a
   conservative (smaller) step. Safe and already documented. No change.

---

## Future work

- **Random cycle spinning.** Ong & Lustig suppress block-boundary artifacts by randomly
  shifting the volume each iteration, applying block SVST, then unshifting, so artifacts
  average out over iterations (Figueiredo & Nowak 2003, *IEEE TIP* 12(8):906–916; see also
  Coifman & Donoho 1995). This code instead uses non-overlapping (exact-prox) or fixed-overlap
  (LLR averaging) patches. Cycle spinning is important for artifact suppression but non-trivial
  to add — it would wrap `patchSVST` with a per-iteration random shift/unshift. Tracked as
  `TODO(cycle-spinning)` in `scripts/reconstruct.jl`.

- **Optional `log` → `log2`** in the `λ_k` formula (`reconstruct.jl` §7) for bit-level fidelity
  to the reference implementation. ⚠ Shifts $\lambda_k$ by ~2–3% and changes future recon
  outputs; existing saved `.mat` files are unaffected.

---

## Code changes made alongside this review (simplification)

- Removed unused code from `src/recon.jl` (~150 lines): `nn_viewshare`, `sense_comb`, the
  per-patch `Vector`-threshold `patchSVST`/`_svst_loop!` variants, and all 2D-input (3-D array)
  patch operators — none were called by any experiment. Pruned the now-orphaned `using FFTW`
  and `using Statistics` and trimmed the module docstring/exports accordingly.
- De-duplicated the unit-patch block-soft-threshold into a single `_unit_block_svst` helper
  shared by both 4-D `patchSVST` methods.
- Documentation: log-base notes in `reconstruct.jl`, `README.md`, `REFERENCES.bib`; the `:fr`
  comment in `mirt_mod.jl`; the cycle-spinning TODO; and section-comment renumbering.

No change was made to the optimizer internals, the GPU memory management, or the numerical
behavior of the default reconstruction path.
