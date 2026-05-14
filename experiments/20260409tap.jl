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
using .Reconstruct

run_recon(
    fn_ksp          = "/StorageRAID/rexfung/20260409tap/recon/caipi_ts_epi_zf.mat",
    fn_smaps        = "/StorageRAID/rexfung/20260409tap/recon/smaps_bart.mat",
    fn_recon_base   = "/StorageRAID/rexfung/20260409tap/recon/mslr/caipi_ts_recon",
    PATCH_SIZES     = [[90, 90, 60], [1, 1, 1]],
    STRIDES         = [[90, 90, 60], [1, 1, 1]],   # half-overlapping
    NITERS          = 200,
    σ1A_PRECOMPUTED = 1.0,
    use_gpu         = true,    # ← set false for CPU
)