module ActivationGLM

#=
activation.jl
Barebones task-activation (GLM t-map) for quick comparison of reconstructions.

This is a deliberately minimal, dependency-free port of the GLM core. It mirrors
the math of the authoritative `fmri-analysis` repo (double-gamma HRF → design
matrix → OLS contrast t-map) but:
  - uses a Lanczos `gamma` so no SpecialFunctions dependency,
  - uses a fixed-t cutoff so no Distributions dependency,
  - uses an intensity-threshold brain mask so no FSL/BET dependency,
  - produces only scalar summaries + montage-ready t-volumes (no NIfTI export).

It is meant as a fast in-repo arbiter for ranking recons (the metric that
correctly penalizes the over-regularization that inflates tSNR). For
publication-grade maps — BET masking, FDR/Bonferroni correction, NIfTI/fsleyes
export, orthogonal slice views — use the `fmri-analysis` repo.

Rex Fung, University of Michigan
=#

export canonical_hrf, build_design_matrix, fit_glm, compute_tscores,
       simple_brain_mask, activation_tmap, activation_summary

using Statistics: mean, std, quantile
using LinearAlgebra: dot, inv
using FFTW: fft, ifft


# ──────────────────────────────────────────────────────────────────────────────
# Gamma function (Lanczos approximation) — avoids a SpecialFunctions dependency
# ──────────────────────────────────────────────────────────────────────────────

const _LANCZOS_G = 7
const _LANCZOS_C = (
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
)

"""Gamma function via the Lanczos approximation (g=7). Accurate to ~1e-13 for x>0."""
function _gamma(x::Real)
    x = float(x)
    if x < 0.5
        return π / (sin(π * x) * _gamma(1 - x))   # reflection formula
    end
    x -= 1
    a = _LANCZOS_C[1]
    t = x + _LANCZOS_G + 0.5
    for i in 2:length(_LANCZOS_C)
        a += _LANCZOS_C[i] / (x + (i - 1))
    end
    return sqrt(2π) * t^(x + 0.5) * exp(-t) * a
end


# ──────────────────────────────────────────────────────────────────────────────
# 1. Haemodynamic response function (SPM-style double-gamma)
# ──────────────────────────────────────────────────────────────────────────────

"""
    canonical_hrf(tr; peak=6.0, undershoot=16.0, peak_disp=1.0,
                  undershoot_disp=1.0, ratio=0.167)

Double-gamma canonical HRF sampled at `tr` seconds, normalized to unit peak,
covering 32 s. Matches the `fmri-analysis` reference implementation.
"""
function canonical_hrf(tr::Real;
    peak::Real=6.0, undershoot::Real=16.0,
    peak_disp::Real=1.0, undershoot_disp::Real=1.0, ratio::Real=0.167)

    t = 0:tr:32.0
    gamma_pdf(t, a, b) = t^(a - 1) * exp(-t / b) / (b^a * _gamma(a))
    h = [gamma_pdf(ti, peak / peak_disp, peak_disp) -
         ratio * gamma_pdf(ti, undershoot / undershoot_disp, undershoot_disp)
         for ti in t]
    return h ./ maximum(abs.(h))
end


# ──────────────────────────────────────────────────────────────────────────────
# 2. Design matrix (FFT-convolved boxcar × HRF, + intercept)
# ──────────────────────────────────────────────────────────────────────────────

"""FFT-based 1-D convolution — O(n log n)."""
function _conv(u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    n = length(u) + length(v) - 1
    N = nextpow(2, n)
    U = fft([Float64.(u); zeros(N - length(u))])
    V = fft([Float64.(v); zeros(N - length(v))])
    return real(ifft(U .* V))[1:n]
end

"""
    build_design_matrix(onsets, durations, n_scans, tr)

`(n_scans × n_conditions+1)` design matrix; last column is the intercept.
`onsets`/`durations` are per-condition vectors in seconds. Stimulus boxcars are
built at 16× temporal oversampling, convolved with the HRF, then downsampled.
"""
function build_design_matrix(
    onsets::AbstractVector{<:AbstractVector{<:Real}},
    durations::AbstractVector{<:AbstractVector{<:Real}},
    n_scans::Int, tr::Real)

    oversampling = 16
    dt = tr / oversampling
    hrf_fine = canonical_hrf(dt)
    n_cond = length(onsets)
    X = zeros(n_scans, n_cond + 1)

    for c in 1:n_cond
        n_fine = n_scans * oversampling
        stimulus = zeros(n_fine)
        for (onset, dur) in zip(onsets[c], durations[c])
            i_start = max(1, round(Int, onset / dt) + 1)
            i_end = min(n_fine, round(Int, (onset + dur) / dt))
            i_start <= i_end && (stimulus[i_start:i_end] .= 1.0)
        end
        convolved = _conv(stimulus, hrf_fine)[1:n_fine]
        X[:, c] = convolved[1:oversampling:end][1:n_scans]
    end
    X[:, end] .= 1.0
    return X
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. OLS fit + contrast t-scores
# ──────────────────────────────────────────────────────────────────────────────

"""OLS fit of `Y = Xβ + ε` (Y is n_scans × n_voxels). Returns `β, residuals, (XᵀX)⁻¹`."""
function fit_glm(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    XtXinv = inv(X' * X)
    beta = XtXinv * (X' * Y)
    residuals = Y - X * beta
    return beta, residuals, XtXinv
end

"""Voxel-wise contrast t-scores: `t = cᵀβ / sqrt(σ² · cᵀ(XᵀX)⁻¹c)`, NaN→0."""
function compute_tscores(beta, residuals, XtXinv, contrast::AbstractVector{<:Real})
    n = size(residuals, 1)
    p = size(beta, 1)
    df = n - p
    sigma2 = max.(vec(sum(residuals .^ 2; dims=1)) ./ df, eps())
    c_var = dot(contrast, XtXinv * contrast)
    c_beta = vec(contrast' * beta)
    t = c_beta ./ sqrt.(sigma2 .* c_var)
    t[isnan.(t)] .= 0.0
    return t
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. Brain mask (intensity threshold — barebones substitute for FSL BET)
# ──────────────────────────────────────────────────────────────────────────────

"""
    simple_brain_mask(mean_vol; frac=0.1)

Crude brain mask: voxels whose temporal-mean magnitude exceeds `frac` × the 99th
percentile (robust max) of the volume. Not anatomically precise — it only needs
to exclude background air so the suprathreshold counts are comparable across
recons. For a real mask use `fmri-analysis`'s `bet_brain_mask`.
"""
function simple_brain_mask(mean_vol::AbstractArray{<:Number,3}; frac::Real=0.1)
    v = Float32.(abs.(mean_vol))
    thr = Float32(frac) * quantile(vec(v), 0.99f0)
    return BitArray(v .> thr)
end


# ──────────────────────────────────────────────────────────────────────────────
# 5. End-to-end: t-map for a 4-D reconstruction + scalar summary
# ──────────────────────────────────────────────────────────────────────────────

"""
    activation_tmap(X4d, paradigm; brain_mask=nothing)

Compute a contrast t-map for a 4-D `(Nx,Ny,Nz,Nt)` reconstruction.
`paradigm` is a NamedTuple with fields
`(tr, onsets, durations, contrast, n_discard)`. Complex input → magnitude.

Returns `(t_vol, brain_mask, df)`.
"""
function activation_tmap(X4d::AbstractArray{<:Number,4}, paradigm; brain_mask=nothing)
    Y = X4d[:, :, :, (paradigm.n_discard + 1):end]
    nx, ny, nz, nt = size(Y)
    mag = eltype(Y) <: Complex ? Float32.(abs.(Y)) : Float32.(Y)

    if brain_mask === nothing
        mean_vol = dropdims(mean(mag; dims=4); dims=4)
        brain_mask = simple_brain_mask(mean_vol)
    end
    bm_flat = vec(brain_mask)

    Ymat = Matrix{Float32}(transpose(reshape(mag, :, nt)))   # (nt × V)
    Ybrain = Ymat[:, bm_flat]

    Xd = build_design_matrix(paradigm.onsets, paradigm.durations, nt, paradigm.tr)
    @assert length(paradigm.contrast) == size(Xd, 2) "contrast length must equal n_regressors"
    beta, residuals, XtXinv = fit_glm(Xd, Ybrain)
    t_brain = compute_tscores(beta, residuals, XtXinv, paradigm.contrast)

    t_flat = zeros(Float32, nx * ny * nz)
    t_flat[bm_flat] .= Float32.(t_brain)
    return reshape(t_flat, nx, ny, nz), brain_mask, nt - size(Xd, 2)
end

"""
    activation_summary(t_vol, brain_mask; t_thresh=5.0)

Scalar activation metrics over in-mask voxels for the positive contrast
(tap > rest): peak t, peak |t|, # and % of voxels with t > `t_thresh`, and the
mean of the top-1% positive t-scores. Returns a NamedTuple.
"""
function activation_summary(t_vol::AbstractArray{<:Real,3}, brain_mask; t_thresh::Real=5.0)
    tb = Float32.(t_vol[brain_mask])
    nvox = length(tb)
    nvox == 0 && return (; peak_t=0f0, peak_abs=0f0, n_supra=0, pct_supra=0f0,
                          mean_top1=0f0, cap=2f0 * Float32(t_thresh), nvox=0,
                          t_thresh=Float32(t_thresh))
    n_supra = count(>(Float32(t_thresh)), tb)
    q99 = quantile(tb, 0.99f0)
    top = tb[tb .>= q99]
    # robust display cap (99.9th pct of |t|) so a few hot voxels don't wash out
    # the colormap; floored at 2× threshold so the scale is always meaningful.
    cap = max(quantile(abs.(tb), 0.999f0), 2f0 * Float32(t_thresh))
    return (; peak_t=maximum(tb), peak_abs=maximum(abs.(tb)),
            n_supra, pct_supra=100f0 * n_supra / nvox,
            mean_top1=isempty(top) ? 0f0 : Float32(mean(top)),
            cap=Float32(cap), nvox, t_thresh=Float32(t_thresh))
end

end # module ActivationGLM
