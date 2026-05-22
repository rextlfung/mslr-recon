#=
20241017tap.jl
Experiment configuration for the 2024-10-17 finger-tapping fMRI dataset.
2.4 mm isotropic, 6× pseudo-random undersampling, Nt=300 frames, 10 virtual coils.

Set use_gpu = true / false below, then run:

  CPU (multi-threaded):
      julia -t auto experiments/20241017tap.jl

  GPU (recommended — RTX A6000 fits a 3-scale run in ~33 GB VRAM):
      julia experiments/20241017tap.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
Revise.includet(joinpath(@__DIR__, "..", "scripts", "analyze.jl"))
Revise.includet(joinpath(@__DIR__, "..", "utils", "recon_cache.jl"))
using .Reconstruct
using .ReconCache

const RECON_DIR   = "/mnt/storage/rexfung/20241017tap"
const FN_SMAPS    = joinpath(RECON_DIR, "smaps_bart.mat")
const PATCH_SIZES       = [[90, 90, 60], [30, 30, 30], [10, 10, 10]]
const STRIDES           = [[90, 90, 60], [30, 30, 30], [10, 10, 10]]
const NITERS            = 50
const σ1A_PRECOMPUTED   = 1.0
const MOM               = :fpgm
const CONV_TOL          = 1e-5

fn_out = joinpath(RECON_DIR, "recon.mat")
try
    if !params_match(fn_out;
            NITERS          = NITERS,
            PATCH_SIZES     = PATCH_SIZES,
            STRIDES         = STRIDES,
            σ1A_PRECOMPUTED = σ1A_PRECOMPUTED,
            mom             = MOM,
            conv_tol        = CONV_TOL)
        println("Reconstructing: rand6x.mat")
        fn_out = run_recon(
            fn_ksp          = joinpath(RECON_DIR, "rand6x.mat"),
            fn_smaps        = FN_SMAPS,
            fn_recon        = fn_out,
            PATCH_SIZES     = PATCH_SIZES,
            STRIDES         = STRIDES,
            NITERS          = NITERS,
            σ1A_PRECOMPUTED = σ1A_PRECOMPUTED,
            mom             = MOM,
            conv_tol        = CONV_TOL,
            use_gpu         = true,    # ← set false for CPU
        )
    end
    run_report(fn_out)
catch e
    @error "Failed on $(basename(fn_out))" exception=(e, catch_backtrace())
end
