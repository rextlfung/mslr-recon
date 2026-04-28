#=
20251106balltap.jl
Experiment configuration for the 2025-11-06 ball phantom + finger-tapping dataset.
2.4 mm isotropic, 6× pseudo-random undersampling, Nt=300 frames, 18 virtual coils.

Set use_gpu = true / false below, then run:

  CPU (multi-threaded):
      julia -t auto experiments/20251106_balltap.jl

  GPU (recommended — RTX A6000 fits a 3-scale run in ~33 GB VRAM):
      julia experiments/20251106_balltap.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Unitful: mm

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
using .Reconstruct

run_recon(
    fn_ksp          = "/StorageRAID/rexfung/20251106balltap/tap/recon/rand6x.mat",
    fn_smaps        = "/StorageRAID/rexfung/20251106balltap/tap/recon/smaps_bart.mat",
    fn_recon_base   = "/StorageRAID/rexfung/20251106balltap/tap/recon/recon",
    N               = (90, 90, 60),
    Nvc             = 18,
    Nt              = 300,
    FOV             = (216mm, 216mm, 144mm),
    PATCH_SIZES     = [[90, 90, 60], [30, 30, 30], [10, 10, 10]],
    STRIDES         = [[90, 90, 60], [30, 30, 30], [10, 10, 10]],
    NITERS          = 50,
    σ1A_PRECOMPUTED = 1.0,
    use_gpu         = true,    # ← set false for CPU
)