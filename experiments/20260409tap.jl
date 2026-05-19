#=
20260409tap.jl
Experiment configuration for the 2026-04-08 finger-tapping dataset.
2.4 mm isotropic, 21 virtual coils, Nt=387 frames.
Multi-scale LR decomposition: global + local + sparse scales, half-overlapping patches.
Runs all 3 acquired datasets sequentially (caipi_ts, caipi, pd).

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
using .Reconstruct

const RECON_DIR  = "/StorageRAID/rexfung/20260409tap/recon"
const FN_SMAPS   = joinpath(RECON_DIR, "smaps_bart.mat")
const PATCH_SIZES = [[90, 90, 60], [6, 6, 6], [1, 1, 1]]
const STRIDES     = [[90, 90, 60], [3, 3, 3], [1, 1, 1]]
const NSCALES     = length(PATCH_SIZES)

datasets = [
    (ksp = "caipi_epi_zf.mat",    base = "mslr/caipi_recon"),
    (ksp = "caipi_ts_epi_zf.mat", base = "mslr/caipi_ts_recon"),
    (ksp = "pd_epi_zf.mat",       base = "mslr/pd_recon"),
]

for ds in datasets
    fn_out = joinpath(RECON_DIR, "$(ds.base)_$(NSCALES)scales.mat")
    println("Reconstructing: $(ds.ksp)")
    if !isfile(fn_out)
        fn_out = run_recon(
            fn_ksp          = joinpath(RECON_DIR, ds.ksp),
            fn_smaps        = FN_SMAPS,
            fn_recon_base   = joinpath(RECON_DIR, ds.base),
            PATCH_SIZES     = PATCH_SIZES,
            STRIDES         = STRIDES,
            NITERS          = 100,
            σ1A_PRECOMPUTED = 0.968294, # measured via tests/sigma1A_test.jl
            use_gpu         = true,
            mom             = :fpgm,
        )
    end
    run_analysis(fn_out)
end