#=
20260409tap.jl
Experiment configuration for the 2026-04-08 finger-tapping dataset.
2.4 mm isotropic, 21 virtual coils, Nt=387 frames.
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
Revise.includet(joinpath(@__DIR__, "..", "scripts", "analyze.jl"))
Revise.includet(joinpath(@__DIR__, "..", "utils", "recon_cache.jl"))
using .Reconstruct
using .ReconCache

const RECON_DIR  = "/StorageRAID/rexfung/20260409tap/recon"
const FN_SMAPS   = joinpath(RECON_DIR, "smaps_bart.mat")
const PATCH_SIZES       = [[90, 90, 60], [6,6,6], [1,1,1]]
const STRIDES           = PATCH_SIZES # non-overlapping patches
const NSCALES           = length(PATCH_SIZES)
const NITERS            = 100 # max number of iterations
const σ1A_PRECOMPUTED   = 1.0 # bounded by construction of encoding operator with unitary FFTs and normalized smaps
const MOM               = :pogm # momentum
const CONV_TOL          = 1e-2 # early stop tolerance for ||x_k - x_(k-1)||/||x_(k-1)||
const λ_SCALE           = 1.0  # multiplicative scale applied to auto-computed regularization weights
const USE_GPU           = false

datasets = [
    (ksp = "caipi_epi_zf.mat",    base = "mslr/caipi_recon"),
    (ksp = "caipi_ts_epi_zf.mat", base = "mslr/caipi_ts_recon"),
    (ksp = "pd_epi_zf.mat",       base = "mslr/pd_recon"),
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
                σ1A_PRECOMPUTED = σ1A_PRECOMPUTED,
                mom             = MOM,
                conv_tol        = CONV_TOL,
                λ_SCALE         = λ_SCALE,
                use_gpu         = USE_GPU,
            )
            run_report(fn_out)
        elseif params_match(fn_out;
                NITERS          = NITERS,
                PATCH_SIZES     = PATCH_SIZES,
                STRIDES         = STRIDES,
                σ1A_PRECOMPUTED = σ1A_PRECOMPUTED,
                mom             = MOM,
                conv_tol        = CONV_TOL,
                lambda_scale    = λ_SCALE)
            run_report(fn_out)
        else
            @warn "Skipping $(ds.ksp): $(fn_out) exists with different parameters — shelve it first."
        end
    catch e
        @error "Failed on $(ds.ksp)" exception=(e, catch_backtrace())
    end
end