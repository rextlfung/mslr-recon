#=
20260810ball.jl
Experiment configuration for the 2026-08-10 ball-phantom dataset.
Two acquisitions with independent spatial grids, coil counts, and smaps files:
  - wb_2.4mm:   whole-brain, 2.4mm iso, 12 virtual coils, Nx=90,Ny=90,Nz=60,   Nt=75
  - slab_0.9mm: slab,        0.9mm iso, 18 virtual coils, Nx=240,Ny=240,Nz=45, Nt=30

Sweeps two axes per dataset: low-rank configuration (local-only vs global+local+sparse,
with the global scale bound to each dataset's own spatial dims) and λ_GLOBAL (reusing
the 1x-10x sweep from 20260317tap.jl). σ1A uses the conservative 1.0 upper bound since
no measured value exists yet for this dataset (verified safe: measured σ1(A)≈0.999 via
power iteration on both wb_2.4mm and slab_0.9mm, all 30 slab frames checked).

Two GPU-specific bugs were found and worked around here — see
mslr/_diverged_gpu_cyclespin/DIAGNOSIS.md for the full writeup:

1. GPU + cycle_spin=true reliably corrupts wb_2.4mm to all-NaN by iteration 3
   (confirmed via a 2x2 CPU/GPU x spin/no-spin isolation). Worked around here with
   CYCLE_SPIN=false (GPU stays correct without spin, and is ~7x faster too).

2. `Asense_gpu` (src/sense_gpu.jl) had a genuine adjoint-correctness bug for ODD
   spatial dimensions: slab_0.9mm's Nz=45 is the first odd dimension ever exercised
   in this codebase (all prior data, including wb_2.4mm, is 90x90x60 — all even).
   Confirmed directly: the <Ax,y> ≈ <x,A'y> identity held to float precision on
   CPU (MIRT.Asense) but was off by 159% on GPU for this shape (fftshift/ifftshift
   swapped in the adjoint — coincide for even N, differ for odd N). FIXED in
   src/sense_gpu.jl on 2026-08-16 and verified: tests/kernel_tests.jl now includes
   an odd-dimension regression case (17/17 pass), and the real slab_0.9mm adjoint
   error dropped from 159% to 0.037% (matching CPU). slab_0.9mm's `use_gpu` below
   was switched back to `true` accordingly — the first 6 "L" config runs (all λ)
   already completed correctly on CPU before the fix; only later runs use GPU.

Set each dataset's `use_gpu` below as needed.

  CPU (multi-threaded):
      julia -t auto experiments/20260810ball.jl

  GPU (recommended for datasets with use_gpu=true):
      julia experiments/20260810ball.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
Revise.includet(joinpath(@__DIR__, "..", "scripts", "reconstruct.jl"))
Revise.includet(joinpath(@__DIR__, "..", "scripts", "report.jl"))
Revise.includet(joinpath(@__DIR__, "..", "utils", "recon_cache.jl"))
using .Reconstruct
using .ReconCache

const RECON_DIR  = "/StorageRAID/rexfung/20260810ball/recon" # CHANGE ME: machine-specific data path

const CYCLE_SPIN = false  # GPU + cycle_spin=true diverges to NaN on wb_2.4mm — see
                           # mslr/_diverged_gpu_cyclespin/DIAGNOSIS.md. Use CPU (24 s/iter,
                           # fastest measured) if the shift-invariance benefit is wanted.
const σ1A        = 1.0    # conservative upper-bound; measured σ1(A)≈0.999 on both datasets
                           # via power iteration (all frames checked for slab), so 1.0 is
                           # confirmed safe (not the cause of either GPU bug documented above)
const NITERS     = 100    # number of iterations
const TOL        = 1e-3   # early stop tolerance for ||x_k - x_(k-1)||/||x_(k-1)||
const MOMENTUM   = :fpgm  # momentum

const LAMBDA_SWEEP = [
    (1.0,  "1xlambda"),
    (2.0,  "2xlambda"),
    (4.0,  "4xlambda"),
    (6.0,  "6xlambda"),
    (8.0,  "8xlambda"),
    (10.0, "10xlambda"),
]

# Each dataset carries its own smaps file, spatial dims (for the "global" patch scale),
# and its own use_gpu — slab_0.9mm MUST stay on CPU until the Asense_gpu odd-dimension
# adjoint bug documented above is fixed.
datasets = [
    (ksp     = "wb_2.4mm_epi_zf.mat",
     smaps   = "smaps_wb_2.4mm_bart.mat",
     name    = "wb_2.4mm_recon",
     dims    = [90, 90, 60],
     use_gpu = true),
    (ksp     = "slab_0.9mm_epi_zf.mat",
     smaps   = "smaps_slab_0.9mm_bart.mat",
     name    = "slab_0.9mm_recon",
     dims    = [240, 240, 45],
     use_gpu = true),  # Asense_gpu odd-dimension adjoint bug fixed in src/sense_gpu.jl
]

# Two low-rank configuration tiers, with the global scale bound to each dataset's own
# (Nx,Ny,Nz). STRIDES = half-overlapping throughout.
function configs_for(dims::Vector{Int})
    half_overlap(ps) = [cld.(p, 2) for p in ps]

    local_ps       = [[6, 6, 6]]
    glob_loc_sp_ps = [dims, [6, 6, 6], [1, 1, 1]]

    return [
        (name = "L",     PATCH_SIZES = local_ps,       STRIDES = half_overlap(local_ps)),
        (name = "G+L+S", PATCH_SIZES = glob_loc_sp_ps, STRIDES = half_overlap(glob_loc_sp_ps)),
    ]
end

for ds in datasets, cfg in configs_for(ds.dims), (λ_GLOBAL, λ_tag) in LAMBDA_SWEEP
    subdir = "$(cfg.name)_$(λ_tag)"
    fn_out = joinpath(RECON_DIR, "mslr", subdir, "$(ds.name).mat")
    try
        if !isfile(fn_out)
            println("Reconstructing: $(ds.ksp)  [$subdir]  (use_gpu=$(ds.use_gpu))")
            fn_out = run_recon(
                fn_ksp          = joinpath(RECON_DIR, ds.ksp),
                fn_smaps        = joinpath(RECON_DIR, ds.smaps),
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
            @warn "Skipping $(ds.ksp) [$subdir]: $(fn_out) exists with different parameters — shelve it first."
        end
    catch e
        @error "Failed on $(ds.ksp) [$subdir]" exception=(e, catch_backtrace())
    end
end
