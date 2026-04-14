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

using Unitful: mm

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
using .Reconstruct

run_recon(
    fn_ksp          = "/StorageRAID/rexfung/20260409tap/recon/caipi_epi_zf.mat",
    fn_smaps        = "/StorageRAID/rexfung/20260409tap/recon/smaps_pisco.mat",
    fn_recon_base   = "/StorageRAID/rexfung/20260409tap/recon/caipi_recon",
    N               = (90, 90, 60),
    Nvc             = 18,
    Nt              = 387,
    FOV             = (216mm, 216mm, 144mm),
    N_gre           = (108, 108, 108),
    FOV_gre         = (216mm, 216mm, 216mm),
    PATCH_SIZES     = [[6, 6, 6]],
    STRIDES         = [[3, 3, 3]],   # half-overlapping
    NITERS          = 50,
    σ1A_PRECOMPUTED = 1.0,
    use_gpu         = false,    # ← set false for CPU
)