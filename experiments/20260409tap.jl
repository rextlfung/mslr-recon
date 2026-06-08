#=
20260409tap.jl
Experiment configuration for the 2026-04-08 finger-tapping dataset.
2.4 mm isotropic, 21 virtual coils, Nt=375 frames.
Multi-scale LR decomposition: global + local + sparse scales, half-overlapping patches.
Runs all 3 acquired datasets sequentially (caipi, caipi_ts, pd).

Set use_gpu = true / false below, then run:

  CPU (multi-threaded):
      julia -t auto experiments/20260409tap.jl

  GPU (recommended — RTX A6000 fits a 3-scale run; see CLAUDE.md for peak VRAM analysis):
      julia experiments/20260409tap.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
Revise.includet(joinpath(@__DIR__, "..", "scripts", "report.jl"))
Revise.includet(joinpath(@__DIR__, "..", "utils", "recon_cache.jl"))
using .Reconstruct
using .ReconCache

const RECON_DIR  = "/StorageRAID/rexfung/20260409tap/recon"
const FN_SMAPS   = joinpath(RECON_DIR, "smaps_bart.mat")
const PATCH_SIZES       = [[90,90,60],[20,20,20],[6,6,6]] # global + local + local
const STRIDES           = [cld.(ps, 2) for ps in PATCH_SIZES] # half-overlapping patches
const NSCALES           = length(PATCH_SIZES) # number of scales
const CYCLE_SPIN        = true # random cycle spin for shift-invariant regularization
const σ1A               = 1.0 # upper-bound from unitary FFTs and normalized smaps
const λ_GLOBAL          = 3.0 # global tuner for regularization weights
const NITERS            = 100 # number of iterations
const TOL               = 1e-3 # early stop tolerance for ||x_k - x_(k-1)||/||x_(k-1)||
const USE_GPU           = true # GPU acceleration
const MOMENTUM          = :fpgm # momentum

# Task paradigm for the optional barebones activation report (tap > rest).
# Acquisition metadata, kept here so the recon repo has no dependency on the
# separate fmri-analysis repo. Mirrors ../fmri-analysis/experiments/20260409tap.jl.
# Set to `nothing` to skip activation and emit the plain 2×2 report.
const PARADIGM = (
    tr        = 0.8f0,
    onsets    = [collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations = [fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast  = [1.0f0, -1.0f0, 0.0f0],   # tap, rest, intercept
    n_discard = 0,                        # these recons are pre-trimmed (Nt=375)
)

datasets = [
    (ksp = "pd_epi_zf.mat",       base = "mslr/pd_recon"),
    (ksp = "caipi_epi_zf.mat",    base = "mslr/caipi_recon"),
    (ksp = "caipi_ts_epi_zf.mat", base = "mslr/caipi_ts_recon"),
]

for ds in datasets
    fn_out = joinpath(RECON_DIR, "$(ds.base).mat")
    try
        if !isfile(fn_out)
            println("Reconstructing: $(ds.ksp)")
            fn_out = run_recon(
                fn_ksp          = joinpath(RECON_DIR, ds.ksp),
                fn_smaps        = FN_SMAPS,
                fn_recon        = fn_out,
                PATCH_SIZES     = PATCH_SIZES,
                STRIDES         = STRIDES,
                NITERS          = NITERS,
                σ1A             = σ1A,
                mom             = MOMENTUM,
                conv_tol        = TOL,
                λ_GLOBAL        = λ_GLOBAL,
                cycle_spin      = CYCLE_SPIN,
                use_gpu         = USE_GPU,
            )
            run_report(fn_out; paradigm = PARADIGM)
        elseif params_match(fn_out;
                NITERS          = NITERS,
                PATCH_SIZES     = PATCH_SIZES,
                STRIDES         = STRIDES,
                σ1A             = σ1A,
                mom             = MOMENTUM,
                conv_tol        = TOL,
                lambda_global   = λ_GLOBAL,
                cycle_spin      = CYCLE_SPIN)
            run_report(fn_out; paradigm = PARADIGM)
        else
            @warn "Skipping $(ds.ksp): $(fn_out) exists with different parameters — shelve it first."
        end
    catch e
        @error "Failed on $(ds.ksp)" exception=(e, catch_backtrace())
    end
end