#=
20260317tap.jl
Experiment configuration for the 2026-03-17 finger-tapping dataset
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

using Unitful: mm

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
using .Reconstruct

# %%
Base.invokelatest(run_recon;
    fn_ksp          = "/mnt/storage/rexfung/20260317tap/recon/pd_epi_zf.mat",
    fn_smaps        = "/mnt/storage/rexfung/20260317tap/recon/smaps_bart.mat",
    fn_recon_base   = "/mnt/storage/rexfung/20260317tap/recon/pd_recon",
    N               = (90, 90, 60),
    Nvc             = 18,
    Nt              = 387,
    FOV             = (216mm, 216mm, 144mm),
    N_gre           = (108, 108, 108),
    FOV_gre         = (216mm, 216mm, 216mm),
    PATCH_SIZES     = [[90, 90, 60], [6, 6, 6], [1, 1, 1]],
    STRIDES         = [[45, 45, 30], [3, 3, 3], [1, 1, 1]], # half-overlapping
    NITERS          = 50,
    σ1A_PRECOMPUTED = 1.0,
    use_gpu         = true # ← set false for CPU
)