#=
20260317tap.jl
Experiment configuration for the 2026-03-17 finger-tapping dataset.
2.4 mm isotropic, 18 virtual coils, Nt=387 frames.
Multi-scale LR decomposition: global + local + sparse scales, half-overlapping patches.

Set use_gpu = true / false below, then run:

  CPU (multi-threaded):
      julia -t auto experiments/20260317tap.jl

  GPU (recommended — RTX A6000 fits a 3-scale run in ~33 GB VRAM):
      julia experiments/20260317tap.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
Revise.includet(joinpath(@__DIR__, "..", "scripts", "analyze.jl"))
Revise.includet(joinpath(@__DIR__, "..", "utils", "recon_cache.jl"))
using .Reconstruct
using .ReconCache

const RECON_DIR   = "/StorageRAID/rexfung/20260317tap/recon"
const FN_SMAPS    = joinpath(RECON_DIR, "smaps_bart.mat")
const PATCH_SIZES       = [[90, 90, 60], [6, 6, 6], [1, 1, 1]]
const STRIDES           = [[45, 45, 30], [3, 3, 3], [1, 1, 1]]   # half-overlapping
const NITERS            = 50
const σ1A               = 1.0
const MOMENTUM          = :fpgm
const TOL               = 1e-5

fn_out = joinpath(RECON_DIR, "caipi_recon.mat")
try
    if !isfile(fn_out)
        println("Reconstructing: caipi_epi_zf.mat")
        fn_out = run_recon(
            fn_ksp          = joinpath(RECON_DIR, "caipi_epi_zf.mat"),
            fn_smaps        = FN_SMAPS,
            fn_recon        = fn_out,
            PATCH_SIZES     = PATCH_SIZES,
            STRIDES         = STRIDES,
            NITERS          = NITERS,
            σ1A             = σ1A,
            mom             = MOMENTUM,
            conv_tol        = TOL,
            use_gpu         = false,    # ← set false for CPU
        )
        run_report(fn_out)
    elseif params_match(fn_out;
            NITERS          = NITERS,
            PATCH_SIZES     = PATCH_SIZES,
            STRIDES         = STRIDES,
            σ1A             = σ1A,
            mom             = MOMENTUM,
            conv_tol        = TOL)
        run_report(fn_out)
    else
        @warn "Skipping caipi_epi_zf.mat: $(fn_out) exists with different parameters — shelve it first."
    end
catch e
    @error "Failed on $(basename(fn_out))" exception=(e, catch_backtrace())
end
