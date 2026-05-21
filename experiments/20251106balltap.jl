#=
20251106balltap.jl
Experiment configuration for the 2025-11-06 ball phantom + finger-tapping dataset.
2.4 mm isotropic, 6× pseudo-random undersampling, Nt=300 frames, 18 virtual coils.

Set use_gpu = true / false below, then run:

  CPU (multi-threaded):
      julia -t auto experiments/20251106balltap.jl

  GPU (recommended — RTX A6000 fits a 3-scale run in ~33 GB VRAM):
      julia experiments/20251106balltap.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
Revise.includet(joinpath(@__DIR__, "..", "scripts", "analyze.jl"))
Revise.includet(joinpath(@__DIR__, "..", "utils", "recon_cache.jl"))
using .Reconstruct
using .ReconCache

const RECON_DIR   = "/StorageRAID/rexfung/20251106balltap/tap/recon"
const FN_SMAPS    = joinpath(RECON_DIR, "smaps_bart.mat")
const PATCH_SIZES = [[90, 90, 60], [30, 30, 30], [10, 10, 10]]
const STRIDES     = [[90, 90, 60], [30, 30, 30], [10, 10, 10]]

fn_out = joinpath(RECON_DIR, "recon.mat")
try
    if !params_match(fn_out;
            NITERS          = 50,
            PATCH_SIZES     = PATCH_SIZES,
            STRIDES         = STRIDES,
            σ1A_PRECOMPUTED = 1.0,
            mom             = :fpgm)
        println("Reconstructing: rand6x.mat")
        fn_out = run_recon(
            fn_ksp          = joinpath(RECON_DIR, "rand6x.mat"),
            fn_smaps        = FN_SMAPS,
            fn_recon        = fn_out,
            PATCH_SIZES     = PATCH_SIZES,
            STRIDES         = STRIDES,
            NITERS          = 50,
            σ1A_PRECOMPUTED = 1.0,
            mom             = :fpgm,
            use_gpu         = true,    # ← set false for CPU
        )
    end
    run_report(fn_out)
catch e
    @error "Failed on $(basename(fn_out))" exception=(e, catch_backtrace())
end
