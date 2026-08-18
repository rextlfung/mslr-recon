#=
reconstruct.jl
Multi-scale Locally Low-Rank (MSLR) fMRI Reconstruction via Decomposition.

Defines module Reconstruct with a single entry point:

    run_recon(; fn_ksp, fn_smaps, fn_recon,
                PATCH_SIZES, STRIDES, NITERS=200,
                σ1A, use_gpu=false, mom=:fpgm, conv_tol=1e-5)

GPU acceleration (use_gpu=true):
  Requires CUDA.jl (add it with: ] add CUDA)
  - Sensitivity maps, k-space, and the sampling mask are moved to the GPU (CuArray).
  - The SENSE encoding operator A is built with Asense_gpu (cuFFT-based).
  - The image X lives on GPU during FISTA; results are brought back to CPU for saving.
  - patchSVST dispatches automatically to sequential CUSOLVER SVDs via Julia type dispatch.
  - pogm_restart (src/mirt_mod.jl) keeps all scalar momentum coefficients in Float32 to prevent
    CuArray{ComplexF32} promotion to Float64, which would double VRAM usage.
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
using Base.Threads
using LinearMapsAA: block_diag, undim
using MIRT: Asense
using Statistics, StatsBase
using MAT, HDF5, NIfTI

include(joinpath(@__DIR__, "..", "src", "recon.jl"))
include(joinpath(@__DIR__, "..", "src", "metrics.jl"))
include(joinpath(@__DIR__, "..", "src", "mirt_mod.jl"))
using .Recon, .Metrics, .MirtMod

# Load GPU support if CUDA.jl is installed.
# If CUDA is not installed, Asense_gpu will be unavailable
# and use_gpu=true will raise a clear error.
const _CUDA_AVAILABLE = try
    using CUDA
    include(joinpath(@__DIR__, "..", "src", "sense_gpu.jl"))
    using .SenseGPU: Asense_gpu
    CUDA.functional()          # true only if a working GPU is present
catch
    false
end

export run_recon

# ── Input loading: dispatch on file extension ──────────────────────────────
# .mat (MATLAB v7.3, HDF5-backed): complex arrays are stored as a real/imag
# struct, and HDF5.jl already restores MATLAB's logical (column-major)
# dimension order on read — MATLAB's HDF5 writer stores dims C-order-reversed
# on disk, and HDF5.jl reverses them back.
# .h5 (row-major writers, e.g. Python/h5py — the sigpy export pipeline):
# complex arrays are stored natively, but the on-disk dimension order already
# equals the logical (numpy) order, so HDF5.jl's read-time reversal must be
# undone with permutedims to recover it.
# Verified against this codebase's actual paired .mat/.h5 exports (identical
# k-space sampling masks and matching smaps shapes after correction).
function _load_array(fn::String, key::String)
    ext = lowercase(splitext(fn)[2])
    raw = h5read(fn, key)
    if ext == ".mat"
        return ComplexF32.([complex(v.real, v.imag) for v in raw])
    elseif ext == ".h5" || ext == ".hdf5"
        return ComplexF32.(permutedims(raw, reverse(1:ndims(raw))))
    else
        error("Unsupported input file extension '$ext' for $fn — expected .mat or .h5")
    end
end

function run_recon(;
    fn_ksp::String,
    fn_smaps::String,
    fn_recon::String,
    PATCH_SIZES::Vector,
    STRIDES::Vector,
    NITERS::Int       = 200,
    σ1A::Union{Float64,Nothing},
    use_gpu::Bool     = false,
    mom::Symbol       = :fpgm,
    conv_tol::Float64 = 1e-5,
    λ_GLOBAL::Float64 = 1.0,
    cycle_spin::Bool  = false,
)
    # ── GPU sanity check & device identification ──────────────────────────────
    if use_gpu
        _CUDA_AVAILABLE || error(
            "use_gpu=true requires CUDA.jl and a working NVIDIA GPU.\n" *
            "Install CUDA.jl with:  julia -e 'using Pkg; Pkg.add(\"CUDA\")'")
        device_str = CUDA.name(CUDA.device())
        println("GPU acceleration enabled  (device: ", device_str, ")")
        println("  Free VRAM: ", round(CUDA.available_memory() / 1e9; digits=1), " GB  /  ",
                round(CUDA.total_memory()     / 1e9; digits=1), " GB total")
    else
        cpu_model = strip(Sys.cpu_info()[1].model)
        device_str = "$cpu_model ($(Threads.nthreads()) threads)"
        println("CPU reconstruction  (device: ", device_str, ")")
        # Prevent OpenBLAS from spawning its own threads inside each @threads worker.
        # Without this, @threads over patches/frames × BLAS internal threads = oversubscription.
        BLAS.set_num_threads(1)
    end

    Nscales = length(PATCH_SIZES)

    # ── 1. Sensitivity maps: load, cast, normalize ───────────────────────────
    println("Loading sensitivity maps …")
    smaps_raw = _load_array(fn_smaps, "smaps")
    smaps_cpu = smaps_raw ./ (sqrt.(sum(abs2.(smaps_raw); dims=4)) .+ eps(Float32))
    println("  Sensitivity maps: ", size(smaps_cpu))


    # ── 2. Load k-space ───────────────────────────────────────────────────────
    println("Loading k-space …")
    ksp0 = _load_array(fn_ksp, "ksp_epi_zf")
    Nx, Ny, Nz, Nvc, Nt = size(ksp0)
    @assert size(smaps_cpu) == (Nx, Ny, Nz, Nvc) "smaps shape $(size(smaps_cpu)) doesn't match k-space dims ($Nx,$Ny,$Nz,$Nvc)"


    # ── 3. Sampling mask and validation ───────────────────────────────────────
    Ω = (ksp0[:, :, :, 1, :] .!= 0)
    R = (Nx * Ny * Nz) / sum(Ω[:, :, :, 1])
    println("Acceleration factor R ≈ ", round(R; digits=2))

    for ic in 2:Nvc
        @assert Ω == (ksp0[:, :, :, ic, :] .!= 0) "Coil $ic has a different sampling pattern"
    end
    for it in 2:Nt
        @assert sum(Ω[:, :, :, it]) == sum(Ω[:, :, :, it-1]) "Frame $it has a different sample count"
    end


    # ── 4. Move data to GPU (if requested) ────────────────────────────────────
    if use_gpu
        println("Moving data to GPU …")
        smaps    = cu(smaps_cpu)   # CuArray{ComplexF32, 4}
        ksp0_gpu = cu(ksp0)        # CuArray{ComplexF32, 5}
        Ω_idx    = cu(Ω)           # CuArray{Bool, 4} — keeps k-space indexing on GPU
        println("  smaps on GPU: ", typeof(smaps))
    else
        smaps    = smaps_cpu
        ksp0_gpu = ksp0
        Ω_idx    = Ω
    end


    # ── 5. SENSE encoding operator A ──────────────────────────────────────────
    println("Building encoding operator …")
    if use_gpu
        # GPU-native SENSE operator: closures capture cu(smaps), use cuFFT.
        # Asense_gpu internally converts the CPU Bool samp mask to CuArray indices.
        Aframe = (Ω_t, S) -> Asense_gpu(Ω_t, S; fft_forward=true, unitary=true)
    else
        # CPU SENSE operator from MIRT
        Aframe = (Ω_t, S) -> Asense(Ω_t, S; fft_forward=true, unitary=true)
    end
    # Build per-frame operators first. On CPU, Aframes is also used for threaded
    # dc_cost / dc_cost_grad (one operator per Julia thread, no shared mutable state).
    Aframes = [Aframe(Ω[:,:,:,it], smaps) for it in 1:Nt]
    A = block_diag(Aframes...)

    # Flatten k-space to (Nsamples, Nvc, Nt) — discard unsampled zeros.
    # Ω_idx is GPU-resident when use_gpu=true, avoiding scalar indexing on CuArrays.
    ksp_flat = reshape(ksp0_gpu, :, Nvc, Nt)
    ksp = cat([ksp_flat[vec(s), :, it]
               for (it, s) in enumerate(eachslice(Ω_idx; dims=4))]...; dims=3)
    println("  k-space shape after masking: ", size(ksp))

    # ksp0_gpu is no longer needed — drop the reference and reclaim VRAM.
    # For a (90,90,60,21,387) ComplexF32 volume this frees ~30 GB.
    ksp0_gpu = nothing; ksp_flat = nothing; ksp0 = nothing
    if use_gpu
        GC.gc(true); CUDA.reclaim()
        println("  VRAM after freeing ksp0_gpu: free=",
                round(CUDA.available_memory()/1e9; digits=2), " GB")
    end


    # ── 6. Lipschitz constant ─────────────────────────────────────────────────
    if isnothing(σ1A)
        println("Computing σ₁(A) via power iteration (may take ~20 min) …")
        _, σ1A = poweriter(undim(A))
        println("  σ₁(A) = ", round(σ1A; digits=4))
    end
    L = Nscales * σ1A^2


    # ── 7. Regularization weights (Ong & Lustig 2016) ─────────────────────────
    # λ_k = √p_k + √Nt + √(log(N_vox·Nt / max(p_k, Nt))),  p_k = voxels per patch.
    # This is Ong & Lustig eq. (4), √m + √n + √(log(MN/max(m,n))), with the (space × time)
    # block being p_k × Nt so M·N = N_vox·Nt. The log ARGUMENT matches Ong's reference code
    # exactly (their bs·min(m,n) = MN/max(m,n)). The paper states the weight only up to a
    # constant ("~"), so the log BASE is unspecified: we use natural log, whereas Ong's MATLAB
    # reference uses log2 (≈1.20× larger third term ⇒ ~2–3% higher λ overall). λ_GLOBAL absorbs
    # any global rescaling, so natural log is consistent with the paper; switch to log2 only if
    # you want bit-level fidelity to the reference implementation.
    # Formula assumes unit-variance noise in image space. BART prewhitening gives
    # σ_ksp ≈ 1 and A is approximately unitary, so σ_image ≈ 1 — no correction needed.
    N_voxels = Nx * Ny * Nz
    λs = Float32[
        sqrt(prod(PATCH_SIZES[k])) +
        sqrt(Nt) +
        sqrt(log(N_voxels * Nt / max(prod(PATCH_SIZES[k]), Nt)))
        for k in 1:Nscales
    ]
    λs .*= Float32(λ_GLOBAL)
    println("Regularization weights λs = ", round.(λs; digits=6))


    # ── 8. Cost functions and proximal operator ───────────────────────────────
    image_sum(X) = dropdims(sum(X; dims=5); dims=5)

    # On CPU, apply each frame's SENSE operator concurrently. Each Aframes[it] closure
    # owns its own work1/work2 buffers (allocated at Asense construction), so concurrent
    # calls on distinct frame operators are safe. BLAS.set_num_threads(1) above prevents
    # nested parallelism (Julia threads × BLAS internal threads).
    # On GPU the block_diag path is unchanged — CuArrays can't use @threads.
    function dc_cost(X)
        if use_gpu
            return 0.5 * norm(A * image_sum(X) - ksp)^2
        end
        img = image_sum(X)
        res = similar(ksp)
        @threads for it in 1:Nt
            res[:, :, it] .= Aframes[it] * view(img, :,:,:,it) .- ksp[:,:,it]
        end
        return 0.5 * norm(res)^2
    end

    function dc_cost_grad(X)
        if use_gpu
            g = A' * (A * image_sum(X) - ksp)
            return repeat(g; outer=[1, 1, 1, 1, Nscales])
        end
        img = image_sum(X)
        res = similar(ksp)
        @threads for it in 1:Nt
            res[:, :, it] .= Aframes[it] * view(img, :,:,:,it) .- ksp[:,:,it]
        end
        g = similar(img)
        @threads for it in 1:Nt
            g[:,:,:,it] .= Aframes[it]' * view(res, :,:,it)
        end
        return repeat(g; outer=[1, 1, 1, 1, Nscales])
    end

    function reg_cost(X)
        return sum(
            λs[k] * patch_nucnorm(img2patches(view(X, :, :, :, :, k),
                                               PATCH_SIZES[k], STRIDES[k]))
            for k in 1:Nscales
        )
    end

    # patchSVST returns (thresholded_img, nuclear_norm) — the nuclear norm is the sum
    # of thresholded singular values, a free byproduct of the SVD already computed.
    # X[:,:,:,:,k] without @views creates an Array copy (CPU) or CuArray copy (GPU),
    # ensuring dispatch reaches the multi-threaded CPU path (not the streaming GPU path).
    #
    # cycle_spin=true: randomly shift each scale's volume before patchSVST and unshift
    # afterward (Figueiredo & Nowak 2003, IEEE TIP 12(8):906-916; Coifman & Donoho 1995).
    # Skipped for unit patches [1,1,1] — SVST is separable per voxel there, so the shift
    # is a provable no-op. Non-deterministic; random shifts inflate rel_change, potentially
    # preventing early stopping (rel_change reflects shift-to-shift variability).

    # Pre-allocate one overlap-count buffer per scale (reused each g_prox call).
    # similar(smaps, ...) produces a CuArray on GPU or Array on CPU.
    pcounts = [fill!(similar(smaps, Float32, Nx, Ny, Nz), 0f0) for _ in 1:Nscales]

    g_prox = (X, c) -> begin
        # Force GC before the first scale's patch-tensor allocation to free any
        # lingering gradient intermediates from dc_cost_grad (image_sum, Ax, residual, g
        # each up to 4.9 GB on GPU but not collected promptly by the GC).
        use_gpu && (GC.gc(true); CUDA.reclaim())
        reg = 0f0
        for k in 1:Nscales
            img_k = X[:, :, :, :, k]
            psx, psy, psz = PATCH_SIZES[k]
            if cycle_spin && !(psx == 1 && psy == 1 && psz == 1)
                shift = (rand(0:Nx-1), rand(0:Ny-1), rand(0:Nz-1))
                img_k = circshift(img_k, (shift..., 0))
                result, cost = patchSVST(img_k, c * λs[k], PATCH_SIZES[k], STRIDES[k]; pcount=pcounts[k])
                result = circshift(result, (-shift[1], -shift[2], -shift[3], 0))
            else
                result, cost = patchSVST(img_k, c * λs[k], PATCH_SIZES[k], STRIDES[k]; pcount=pcounts[k])
            end
            @views X[:, :, :, :, k] = result
            reg += λs[k] * cost
        end
        last_reg[] = reg
        return X
    end

    # ── 9. Initialize X ──────────────────────────────────────────────────────
    Atksp = (A' * ksp) ./ Nscales
    X0 = repeat(Atksp, outer=(1, 1, 1, 1, Nscales))
    # A'*ksp leaves block-adjoint intermediates (one per time frame) GC-eligible but not yet
    # collected. Reclaim now so they don't inflate VRAM headroom during POGM.
    use_gpu && (GC.gc(true); CUDA.reclaim())


    # ── 10. FISTA ─────────────────────────────────────────────────────────────
    # Gradient restart (:gr) decides restarts from gradient direction, not Fcost values,
    # so passing dc_cost as Fcost avoids a separate cost evaluation each iteration.
    # reg_cost is logged for free via g_prox (sum of thresholded singular values).
    backend_str = use_gpu ? "GPU" : "CPU"
    if use_gpu
        free_b, total_b = CUDA.available_memory(), CUDA.total_memory()
        println("  VRAM before FISTA: free=", round(free_b/1e9; digits=2),
                " GB / total=", round(total_b/1e9; digits=2), " GB")
    end
    if cycle_spin && conv_tol > 0
        @warn "cycle_spin=true with conv_tol=$conv_tol: random shifts inflate rel_change, potentially preventing early stopping. Set conv_tol=0 to disable."
    end
    println("\nIteratively reconstructing on $backend_str ($NITERS iterations, $Nscales scale(s), mom=$mom, conv_tol=$conv_tol, cycle_spin=$cycle_spin) …")
    # reg_cost at iter 0 (before any prox): computed once from the initial iterate.
    last_reg = Ref(reg_cost(use_gpu ? Array(X0) : X0))
    # From iter 1 onward, last_reg[] is updated for free inside g_prox — no extra SVDs.
    # For :fpgm, the captured value is reg_cost(ynew) (prox output), consistent with
    # Fcostnew = dc_cost(ynew); for :pogm/:pgm, ynew === xnew so the value is exact.
    logger = (iter, xk, _, is_restart, Fcostnew, rel_change) ->
        (Fcostnew, last_reg[], is_restart, rel_change)
    t_start = time()
    X, costs = pogm_restart(
        X0, dc_cost, dc_cost_grad, L;
        mom      = mom,
        niter    = NITERS,
        g_prox   = g_prox,
        fun      = logger,
        conv_tol = conv_tol,
    )
    runtime_s    = time() - t_start
    iter_time_s  = runtime_s / (length(costs) - 1)

    dc_costs    = [c[1] for c in costs]
    reg_costs   = [c[2] for c in costs]
    restarts    = [c[3] for c in costs]
    rel_changes = [c[4] for c in costs]
    X_recon     = image_sum(X)

    # Move results back to CPU for saving
    if use_gpu
        println("Moving results back to CPU …")
        X       = Array(X)
        X_recon = Array(X_recon)
    end


    # ── 11. Save ──────────────────────────────────────────────────────────────
    fn_out = fn_recon
    mkpath(dirname(fn_out))
    matwrite(fn_out, Dict(
        "X"            => X,
        "X_recon"      => X_recon,
        "omega"        => Ω,
        "dc_costs"     => dc_costs,
        "reg_costs"    => reg_costs,
        "restarts"     => restarts,
        "rel_changes"  => rel_changes,
        "R"            => R,
        "sigma1A"      => σ1A,
        "L"            => L,
        "Nscales"      => Nscales,
        "patch_sizes"  => PATCH_SIZES,
        "strides"      => STRIDES,
        "lambdas"      => λs,
        "lambda_global" => λ_GLOBAL,
        "Niters"       => NITERS,
        "used_gpu"     => use_gpu,
        "device"       => device_str,
        "mom"          => String(mom),
        "conv_tol"     => conv_tol,
        "cycle_spin"   => cycle_spin,
        "runtime_s"    => runtime_s,
        "iter_time_s"  => iter_time_s,
    ); compress=true)

    # Magnitude NIfTI export (fsleyes/SPM/AFNI-compatible; complex phase is discarded).
    fn_nii = splitext(fn_out)[1] * ".nii.gz"
    niwrite(fn_nii, NIVolume(Float32.(abs.(X_recon))))

    mm, ss = divrem(round(Int, runtime_s), 60)
    println("Wall-clock: $(mm)m $(ss)s ($(round(runtime_s; digits=1)) s), $(round(iter_time_s; digits=1)) s/iter")
    println("\n✓ Saved → $fn_out")
    println("✓ Saved → $fn_nii")
    return fn_out
end

end # module Reconstruct