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
using .Reconstruct

fn_out = "/StorageRAID/rexfung/20260317tap/recon/caipi_recon_3scales.mat"

if !isfile(fn_out)
    fn_out = run_recon(
        fn_ksp          = "/StorageRAID/rexfung/20260317tap/recon/caipi_epi_zf.mat",
        fn_smaps        = "/StorageRAID/rexfung/20260317tap/recon/smaps_bart.mat",
        fn_recon_base   = "/StorageRAID/rexfung/20260317tap/recon/caipi_recon",
        PATCH_SIZES     = [[90, 90, 60], [6, 6, 6], [1, 1, 1]],
        STRIDES         = [[45, 45, 30], [3, 3, 3], [1, 1, 1]],   # half-overlapping
        NITERS          = 50,
        σ1A_PRECOMPUTED = 1.0,
        use_gpu         = false,    # ← set false for CPU
        mom             = :fpgm,
    )
end

run_report(fn_out)