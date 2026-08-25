#=
20260822ball.jl
Experiment configuration for the 2026-08-22 ball-phantom dataset.

Compares the two mask2epi trajectory variants (radial vs. laminar) from
ArbEPI-python's lib/mask2epi.py on otherwise identical acquisitions: 0.9mm
iso, Nx=240,Ny=240,Nz=45, Nt=30 -- same acquisition matrix as 20260810ball's
slab_0.9mm, so its σ1A/λ_GLOBAL/use_gpu precedent applies directly here (see
below). Unlike 20260810ball (one exam, several resolutions sharing one
RECON_DIR), radial and laminar here are separate exams/directories, so each
dataset below carries its own `recon_dir` rather than sharing a single
top-level constant. Uses the sigpy .h5 exports (k-space and smaps) from
ArbEPI-python's preprocessing pipeline.

σ1A: measured directly against this dataset (radial σ1(A)=0.999679, laminar
σ1(A)=0.999681 -- essentially identical, as expected: both trajectory
variants share the same underlying sampling mask/seed, only the per-shot
traversal order differs). Uses the max of the two, 0.999681, as one shared
conservative value for both entries below. Measured via a one-off adaptation
of tests/sigma1A_tests.jl for .h5 (not .mat) input -- see
scratchpad/sigma1A_ball.jl in the ArbEPI-python session that produced this
file; tests/sigma1A_tests.jl itself still only handles .mat.

GPU notes carried over from 20260810ball.jl (mslr/_diverged_gpu_cyclespin/
DIAGNOSIS.md):
1. GPU + cycle_spin=true still reliably diverges (confirmed on wb_2.4mm; no
   fix has landed since 20260810ball.jl was written — checked git log).
   Worked around here with CYCLE_SPIN=false, same as that experiment.
2. Asense_gpu's odd-dimension adjoint bug (Nz=45 is odd) WAS fixed in
   src/sense_gpu.jl on 2026-08-16 and verified against slab_0.9mm's exact
   (240,240,45) shape -- both datasets here share that shape, so `use_gpu =
   true` is safe on the same basis 20260810ball.jl already established.

Set each dataset's `use_gpu` below as needed.

  CPU (multi-threaded):
      julia -t auto experiments/20260822ball.jl

  GPU (recommended for datasets with use_gpu=true):
      julia experiments/20260822ball.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
Revise.includet(joinpath(@__DIR__, "..", "scripts", "report.jl"))
Revise.includet(joinpath(@__DIR__, "..", "utils", "recon_cache.jl"))
using .Reconstruct
using .ReconCache

const CYCLE_SPIN = false  # GPU + cycle_spin=true diverges to NaN — see
                           # mslr/_diverged_gpu_cyclespin/DIAGNOSIS.md. Use CPU (24 s/iter,
                           # fastest measured) if the shift-invariance benefit is wanted.
const σ1A        = 0.999681  # measured directly (see header): radial=0.999679, laminar=0.999681
const NITERS     = 100    # number of iterations
const TOL        = 1e-3   # early stop tolerance for ||x_k - x_(k-1)||/||x_(k-1)||
const MOMENTUM   = :fpgm  # momentum

const λ_GLOBAL = 9.0  # matches 20260810ball's slab_0.9mm (same Nx,Ny,Nz=240,240,45)

# Radial and laminar trajectory variants, each its own exam/directory, using
# the sigpy .h5 exports (k-space and smaps) ArbEPI-python's preprocessing
# pipeline writes as recon/ArbEPI_epi_zf.h5 / recon/smaps_ArbEPI_sigpy.h5.
datasets = [
    (recon_dir = "/StorageRAID/rexfung/20260822ball_radial/recon",  # CHANGE ME: machine-specific data path
     ksp       = "ArbEPI_epi_zf.h5",
     smaps     = "smaps_ArbEPI_sigpy.h5",
     name      = "radial_recon",
     dims      = [240, 240, 45],
     use_gpu   = true),  # Asense_gpu odd-dimension adjoint bug fixed in src/sense_gpu.jl
    (recon_dir = "/StorageRAID/rexfung/20260822ball_laminar/recon",  # CHANGE ME: machine-specific data path
     ksp       = "ArbEPI_epi_zf.h5",
     smaps     = "smaps_ArbEPI_sigpy.h5",
     name      = "laminar_recon",
     dims      = [240, 240, 45],
     use_gpu   = true),
]

# Three low-rank configuration tiers, with the global scale bound to each dataset's own
# (Nx,Ny,Nz) and the local scale at patch [15,15,15]. STRIDES = half-overlapping throughout.
function configs_for(dims::Vector{Int})
    half_overlap(ps) = [cld.(p, 2) for p in ps]

    glob_ps     = [dims]
    local_ps    = [[15, 15, 15]]
    glob_loc_ps = [dims, [15, 15, 15]]

    return [
        (name = "G",   PATCH_SIZES = glob_ps,     STRIDES = half_overlap(glob_ps)),
        (name = "L",   PATCH_SIZES = local_ps,    STRIDES = half_overlap(local_ps)),
        (name = "G+L", PATCH_SIZES = glob_loc_ps, STRIDES = half_overlap(glob_loc_ps)),
    ]
end

for ds in datasets, cfg in configs_for(ds.dims)
    subdir = cfg.name
    fn_out = joinpath(ds.recon_dir, "mslr", subdir, "$(ds.name).mat")
    try
        if !isfile(fn_out)
            println("Reconstructing: $(ds.name)  [$subdir]  (use_gpu=$(ds.use_gpu))")
            fn_out = run_recon(
                fn_ksp          = joinpath(ds.recon_dir, ds.ksp),
                fn_smaps        = joinpath(ds.recon_dir, ds.smaps),
                fn_recon        = fn_out,
                PATCH_SIZES     = cfg.PATCH_SIZES,
                STRIDES         = cfg.STRIDES,
                NITERS          = NITERS,
                σ1A             = σ1A,
                mom             = MOMENTUM,
                conv_tol        = TOL,
                λ_GLOBAL        = λ_GLOBAL,
                cycle_spin      = CYCLE_SPIN,
                use_gpu         = ds.use_gpu,
            )
            run_report(fn_out)
        elseif params_match(fn_out;
                NITERS          = NITERS,
                PATCH_SIZES     = cfg.PATCH_SIZES,
                STRIDES         = cfg.STRIDES,
                σ1A             = σ1A,
                mom             = MOMENTUM,
                conv_tol        = TOL,
                lambda_global   = λ_GLOBAL,
                cycle_spin      = CYCLE_SPIN)
            println("Already done: $subdir/$(ds.name) — regenerating report")
            run_report(fn_out)
        else
            @warn "Skipping $(ds.name) [$subdir]: $(fn_out) exists with different parameters — shelve it first."
        end
    catch e
        @error "Failed on $(ds.name) [$subdir]" exception=(e, catch_backtrace())
    end
end
