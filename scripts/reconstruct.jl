#=
reconstruct.jl
Multi-scale Locally Low-Rank (MSLR) fMRI Reconstruction via Decomposition.

Defines module Reconstruct with a single entry point:

    run_recon(; fn_ksp, fn_smaps, fn_recon_base, N, Nvc, Nt, FOV,
                PATCH_SIZES, STRIDES, NITERS,
                σ1A_PRECOMPUTED, use_gpu=false)

GPU acceleration (use_gpu=true):
  Requires CUDA.jl (add it with: ] add CUDA)
  - Sensitivity maps and k-space are moved to the GPU (CuArray).
  - The SENSE encoding operator A is built with Asense_gpu (cuFFT-based).
  - The image X lives on GPU during POGM; results are brought back to CPU for saving.
  - patchSVST_gpu is used for the proximal step (sequential CUSOLVER SVDs).
  - RTX A6000 (48 GB) fits a 3-scale run (~33 GB) comfortably.

Algorithm:
  X_final = X[:,:,:,:,1] + ... + X[:,:,:,:,Nscales]

  Each component X[:,:,:,:,k] is independently constrained to be locally
  low-rank at its own patch scale. Data consistency is enforced on the sum.
  λ_k set by the Ong & Lustig (2016) formula (theoretically motivated, no tuning).

Rex Fung, University of Michigan
Based on main9993.jl
=#

module Reconstruct

using LinearAlgebra
using LinearMapsAA: block_diag, undim
using MIRT: Asense
using Statistics, StatsBase
using ProgressMeter
using MAT, HDF5
using Unitful: mm

include(joinpath(@__DIR__, "..", "src", "recon.jl"))
include(joinpath(@__DIR__, "..", "src", "mirt_mod.jl"))
include(joinpath(@__DIR__, "..", "src", "analysis.jl"))
using .Recon, .MirtMod, .Analysis

# Load GPU support if CUDA.jl is installed.
# If CUDA is not installed, Asense_gpu / patchSVST_gpu will be unavailable
# and use_gpu=true will raise a clear error.
const _CUDA_AVAILABLE = try
    using CUDA
    include(joinpath(@__DIR__, "..", "src", "sense_gpu.jl"))
    using .SenseGPU
    CUDA.functional()          # true only if a working GPU is present
catch
    false
end

export run_recon

function run_recon(;
    fn_ksp::String,
    fn_smaps::String,
    fn_recon_base::String,
    N::Tuple{Int,Int,Int},
    Nvc::Int,
    Nt::Int,
    FOV::Tuple,
    PATCH_SIZES::Vector,
    STRIDES::Vector,
    NITERS::Int,
    σ1A_PRECOMPUTED::Union{Float64,Nothing},
    use_gpu::Bool = false,
)
    # ── GPU sanity check ──────────────────────────────────────────────────────
    if use_gpu
        _CUDA_AVAILABLE || error(
            "use_gpu=true requires CUDA.jl and a working NVIDIA GPU.\n" *
            "Install CUDA.jl with:  julia -e 'using Pkg; Pkg.add(\"CUDA\")'")
        println("GPU acceleration enabled  (device: ", CUDA.name(CUDA.device()), ")")
        println("  Free VRAM: ", round(CUDA.available_memory() / 1e9; digits=1), " GB  /  ",
                round(CUDA.total_memory()     / 1e9; digits=1), " GB total")
    end

    Nx, Ny, Nz = N
    Nscales    = length(PATCH_SIZES)

    # ── 1. Sensitivity maps: load, cast, normalise ───────────────────────────
    println("Loading sensitivity maps …")
    smaps_raw = ComplexF32.(matread(fn_smaps)["smaps"])
    smaps_cpu = smaps_raw ./ (sqrt.(sum(abs2.(smaps_raw); dims=4)) .+ eps(Float32))
    println("  Sensitivity maps: ", size(smaps_cpu))


    # ── 2. Load k-space ───────────────────────────────────────────────────────
    println("Loading k-space …")
    ksp0 = h5read(fn_ksp, "ksp_epi_zf")
    ksp0 = ComplexF32.([complex(k.real, k.imag) for k in ksp0])
    @assert size(ksp0) == (Nx, Ny, Nz, Nvc, Nt) "Unexpected k-space shape: $(size(ksp0))"


    # ── 3. Normalise k-space ──────────────────────────────────────────────────
    println("Normalising k-space …")
    img0         = sense_comb(ksp0, smaps_cpu)
    scale_factor = quantile(vec(abs.(img0)), 0.99)
    ksp0        ./= max(scale_factor, eps(Float32))
    println("  Scale factor = ", round(scale_factor; digits=4))


    # ── 4. Sampling mask and validation ───────────────────────────────────────
    Ω = (ksp0[:, :, :, 1, :] .!= 0)
    R = prod(N) / sum(Ω[:, :, :, 1])
    println("Acceleration factor R ≈ ", round(R; digits=2))

    for ic in 2:Nvc
        @assert Ω == (ksp0[:, :, :, ic, :] .!= 0) "Coil $ic has a different sampling pattern"
    end
    for it in 2:Nt
        @assert sum(Ω[:, :, :, it]) == sum(Ω[:, :, :, it-1]) "Frame $it has a different sample count"
    end


    # ── 5. Move data to GPU (if requested) ────────────────────────────────────
    if use_gpu
        println("Moving data to GPU …")
        smaps = cu(smaps_cpu)      # CuArray{ComplexF32, 4}
        ksp0_gpu = cu(ksp0)        # CuArray{ComplexF32, 5}
        println("  smaps on GPU: ", typeof(smaps))
    else
        smaps    = smaps_cpu
        ksp0_gpu = ksp0
    end


    # ── 6. SENSE encoding operator A ──────────────────────────────────────────
    println("Building encoding operator …")
    if use_gpu
        # GPU-native SENSE operator: closures capture cu(smaps), use cuFFT
        Aframe = (Ω_t, S) -> Asense_gpu(Ω_t, S; fft_forward=true, unitary=true)
    else
        # CPU SENSE operator from MIRT
        Aframe = (Ω_t, S) -> Asense(Ω_t, S; fft_forward=true, unitary=true)
    end
    A = block_diag([Aframe(s, smaps) for s in eachslice(Ω; dims=ndims(Ω))]...)

    # Flatten k-space to (Nsamples, Nvc, Nt) — discard unsampled zeros
    ksp_flat = reshape(ksp0_gpu, :, Nvc, Nt)
    ksp = cat([ksp_flat[vec(s), :, it]
               for (it, s) in enumerate(eachslice(Ω; dims=4))]...; dims=3)
    println("  k-space shape after masking: ", size(ksp))


    # ── 7. Lipschitz constant ─────────────────────────────────────────────────
    if isnothing(σ1A_PRECOMPUTED)
        println("Computing σ₁(A) via power iteration (may take ~20 min) …")
        _, σ1A = poweriter_mod(undim(A))
        println("  σ₁(A) = ", round(σ1A; digits=4))
    else
        σ1A = σ1A_PRECOMPUTED
    end
    L = Nscales * σ1A^2


    # ── 8. Regularisation weights (Ong & Lustig 2016) ─────────────────────────
    N_voxels = prod(N)
    λs = [
        sqrt(prod(PATCH_SIZES[k])) +
        sqrt(Nt) +
        sqrt(log(N_voxels * Nt / max(prod(PATCH_SIZES[k]), Nt)))
        for k in 1:Nscales
    ]
    println("Regularisation weights λs = ", round.(λs; digits=3))


    # ── 9. Cost functions and proximal operator ───────────────────────────────
    image_sum(X) = dropdims(sum(X; dims=5); dims=5)

    function dc_cost(X)
        return 0.5 * norm(A * image_sum(X) - ksp)^2
    end

    function dc_cost_grad(X)
        g = A' * (A * image_sum(X) - ksp)
        return repeat(g; outer=[1, 1, 1, 1, Nscales])
    end

    function reg_cost(X)
        return sum(
            λs[k] * patch_nucnorm(img2patches(view(X, :, :, :, :, k),
                                               PATCH_SIZES[k], STRIDES[k]))
            for k in 1:Nscales
        )
    end

    # Select patch proximal operator based on backend
    _patch_svst = use_gpu ? patchSVST_gpu : patchSVST

    g_prox = (X, c) -> begin
        for k in 1:Nscales
            @views X[:, :, :, :, k] = _patch_svst(
                X[:, :, :, :, k], c * λs[k], PATCH_SIZES[k], STRIDES[k])
        end
        return X
    end

    comp_cost = X -> dc_cost(X) + reg_cost(X)
    logger    = (iter, xk, yk, is_restart) -> (dc_cost(xk), reg_cost(xk), is_restart)


    # ── 10. Initialise X ──────────────────────────────────────────────────────
    if use_gpu
        X0 = CUDA.zeros(ComplexF32, Nx, Ny, Nz, Nt, Nscales)
    else
        X0 = zeros(ComplexF32, Nx, Ny, Nz, Nt, Nscales)
    end
    X0[:, :, :, :, 1] = A' * ksp


    # ── 11. POGM ──────────────────────────────────────────────────────────────
    backend_str = use_gpu ? "GPU" : "CPU"
    println("\nRunning POGM ($NITERS iterations, $Nscales scale(s), $backend_str) …")
    X, costs = pogm_mod(
        X0, comp_cost, dc_cost_grad, L;
        mom    = :pogm,
        niter  = NITERS,
        g_prox = g_prox,
        fun    = logger,
    )

    dc_costs  = [c[1] for c in costs]
    reg_costs = [c[2] for c in costs]
    restarts  = [c[3] for c in costs]
    X_recon   = image_sum(X)

    # Move results back to CPU for saving
    if use_gpu
        println("Moving results back to CPU …")
        X       = Array(X)
        X_recon = Array(X_recon)
    end


    # ── 12. Save ──────────────────────────────────────────────────────────────
    fn_out = fn_recon_base * "_$(Nscales)scales.mat"
    matwrite(fn_out, Dict(
        "X"            => X,
        "X_recon"      => X_recon,
        "omega"        => Ω,
        "dc_costs"     => dc_costs,
        "reg_costs"    => reg_costs,
        "restarts"     => restarts,
        "R"            => R,
        "sigma1A"      => σ1A,
        "L"            => L,
        "Nscales"      => Nscales,
        "patch_sizes"  => PATCH_SIZES,
        "strides"      => STRIDES,
        "lambdas"      => λs,
        "Niters"       => NITERS,
        "scale_factor" => scale_factor,
        "used_gpu"     => use_gpu,
    ); compress=true)

    println("\n✓ Saved → $fn_out")
end

end # module Reconstruct