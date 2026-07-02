#=
20260317tap.jl
Experiment configuration for the 2026-03-17 finger-tapping dataset.
2.4 mm isotropic, 18 virtual coils, Nt=387 frames.
Multi-scale LR decomposition: global + local + sparse scales, half-overlapping patches.
Runs all acquired datasets sequentially.

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
Revise.includet(joinpath(@__DIR__, "..", "scripts", "report.jl"))
Revise.includet(joinpath(@__DIR__, "..", "utils", "recon_cache.jl"))
using .Reconstruct
using .ReconCache

const RECON_DIR         = "/StorageRAID/rexfung/20260317tap/recon" # CHANGE ME: machine-specific data path
const FN_SMAPS          = joinpath(RECON_DIR, "smaps_bart.mat")
const PATCH_SIZES       = [[6,6,6]] # local only
const STRIDES           = [cld.(ps, 2) for ps in PATCH_SIZES] # half-overlapping patches
const NSCALES           = length(PATCH_SIZES)
const CYCLE_SPIN        = true # random cycle spin for shift-invariant regularization
const σ1A               = 1.0 # upper-bound from unitary FFTs and normalized smaps
const NITERS            = 100 # number of iterations
const TOL               = 1e-2 # early stop tolerance for ||x_k - x_(k-1)||/||x_(k-1)||
const USE_GPU           = true # GPU acceleration
const MOMENTUM          = :fpgm # momentum

const LAMBDA_SWEEP = [
    (1.0, "L_1xlambda"),
    (2.0, "L_2xlambda"),
    (4.0, "L_4xlambda"),
    (6.0, "L_6xlambda"),
    (8.0, "L_8xlambda"),
    (10.0, "L_10xlambda"),
]

datasets = [
    (ksp = "pd_epi_zf.mat",    name = "pd_recon"),
    (ksp = "caipi_epi_zf.mat", name = "caipi_recon"),
]

for (λ_GLOBAL, subdir) in LAMBDA_SWEEP, ds in datasets
    fn_out = joinpath(RECON_DIR, "mslr", subdir, "$(ds.name).mat")
    try
        if !isfile(fn_out)
            println("Reconstructing: $(ds.ksp)  [λ=$λ_GLOBAL → $subdir]")
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
            run_report(fn_out)
        elseif params_match(fn_out;
                NITERS          = NITERS,
                PATCH_SIZES     = PATCH_SIZES,
                STRIDES         = STRIDES,
                σ1A             = σ1A,
                mom             = MOMENTUM,
                conv_tol        = TOL,
                lambda_global   = λ_GLOBAL,
                cycle_spin      = CYCLE_SPIN)
            println("Already done: $subdir/$(ds.name) — regenerating report")
            run_report(fn_out)
        else
            @warn "Skipping $(ds.ksp): $(fn_out) exists with different parameters — shelve it first."
        end
    catch e
        @error "Failed on $(ds.ksp) [λ=$λ_GLOBAL]" exception=(e, catch_backtrace())
    end
end
