#=
20260409tap.jl
Experiment configuration for the 2026-04-08 finger-tapping dataset.
2.4 mm isotropic, 18 virtual coils, Nt=387 frames.
Multi-scale LR decomposition: global + local + sparse scales, half-overlapping patches.

Set use_gpu = true / false below, then run:

  CPU (multi-threaded):
      julia -t auto experiments/20260409tap.jl

  GPU (recommended — RTX A6000 fits a 3-scale run in ~33 GB VRAM):
      julia experiments/20260409tap.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
Revise.includet(joinpath(@__DIR__, "..", "scripts", "analyze.jl"))
using .Reconstruct

fn_out = "/StorageRAID/rexfung/20260409tap/recon/mslr/caipi_recon_2scales.mat"

if !isfile(fn_out)
    fn_out = run_recon(
        fn_ksp          = "/StorageRAID/rexfung/20260409tap/recon/caipi_epi_zf.mat",
        fn_smaps        = "/StorageRAID/rexfung/20260409tap/recon/smaps_bart.mat",
        fn_recon_base   = "/StorageRAID/rexfung/20260409tap/recon/mslr/caipi_recon",
        PATCH_SIZES     = [[90, 90, 60], [6, 6, 6]],
        STRIDES         = [[90, 90, 60], [3, 3, 3]],
        NITERS          = 100,
        σ1A_PRECOMPUTED = 0.968294, # measured via scripts/verify_sigma1A.jl
        use_gpu         = true,
        mom             = :pogm, # :pogm, :fpgm, :pgm 
    )
end

run_analysis(fn_out)