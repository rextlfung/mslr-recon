#=
reconstruct.jl
Multi-scale Locally Low-Rank (MSLR) fMRI Reconstruction via Decomposition.

GPU acceleration (use_gpu=true):
  Single-GPU (1 device): identical to the original implementation.
  Multi-GPU  (>1 device): uses multigpu.jl / mirt_gpu.jl.

Memory strategy for multi-GPU:
  ksp (4.5 GB) is NEVER moved to GPU 0.  Instead, A'*ksp (Atksp, 1.5 GB) is
  pre-computed once on GPU 0 frame-by-frame (12 MB peak per frame), then ksp
  is freed from CPU.  The gradient inside pogm_mg is computed as
    g_t = A_frames[t]'*(A_frames[t]*x_t) - Atksp_t
  keeping GPU 0 peak memory at ~5 GB instead of ~12 GB.

Rex Fung, University of Michigan
=#

module Reconstruct

using LinearAlgebra
using LinearMapsAA: block_diag, undim
using MIRT: Asense
using Statistics, StatsBase
using ImageTransformations: imresize
using ProgressMeter
using MAT, HDF5
using Unitful: mm

include(joinpath(@__DIR__, "..", "src", "recon.jl"))
include(joinpath(@__DIR__, "..", "src", "mirt_mod.jl"))
include(joinpath(@__DIR__, "..", "src", "analysis.jl"))
using .Recon, .MirtMod, .Analysis

const _CUDA_AVAILABLE = try
    using CUDA
    include(joinpath(@__DIR__, "..", "src", "sense_gpu.jl"))
    include(joinpath(@__DIR__, "..", "src", "mirt_gpu.jl"))
    include(joinpath(@__DIR__, "..", "src", "multigpu.jl"))
    CUDA.functional()
catch
    false
end

# Bring GPU module names into Reconstruct's scope unconditionally.
# If CUDA is unavailable these are no-ops (the modules simply don't exist,
# but use_gpu=false means none of the functions that call them are reached).
if _CUDA_AVAILABLE
    using .SenseGPU, .MirtGPU, .MultiGPU
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
    N_gre::Tuple{Int,Int,Int},
    FOV_gre::Tuple,
    PATCH_SIZES::Vector,
    STRIDES::Vector,
    NITERS::Int,
    σ1A_PRECOMPUTED::Union{Float64,Nothing},
    use_gpu::Bool=false,
)
    # ── GPU detection ─────────────────────────────────────────────────────────
    use_multigpu = false
    gpu_ids = Int[]
    mom_type = :pogm

    if use_gpu
        _CUDA_AVAILABLE || error(
            "use_gpu=true requires CUDA.jl and a working NVIDIA GPU.\n" *
            "Install with:  julia -e 'using Pkg; Pkg.add(\"CUDA\")'")

        gpu_ids = detect_gpus()

        if length(gpu_ids) > 1
            use_multigpu = true
            println("Multi-GPU mode: $(length(gpu_ids)) device(s).")
            println("  GPU 0 is master (SENSE operator + gradient).")
            println("  Momentum type will be selected after data is loaded onto GPU 0.")
        else
            println("Single-GPU mode  (device: ", CUDA.name(CUDA.device()), ")")
            println("  Free VRAM: ",
                round(CUDA.available_memory() / 1e9; digits=1), " GB  /  ",
                round(CUDA.total_memory() / 1e9; digits=1), " GB total")
        end
    end

    Nx, Ny, Nz = N
    Nscales = length(PATCH_SIZES)


    # ── 1. Sensitivity maps ───────────────────────────────────────────────────
    println("Loading sensitivity maps …")
    smaps_raw = matread(fn_smaps)["smaps_raw"]

    function fov_crop_range(N_src, FOV_src, FOV_dst)
        margin = (FOV_src - FOV_dst) / FOV_src / 2 * N_src
        return Int(round(margin + 1)):Int(round(N_src - margin))
    end
    xr = fov_crop_range(N_gre[1], FOV_gre[1], FOV[1])
    yr = fov_crop_range(N_gre[2], FOV_gre[2], FOV[2])
    zr = fov_crop_range(N_gre[3], FOV_gre[3], FOV[3])
    smaps_crop = smaps_raw[xr, yr, zr, :]

    smaps_interp = complex.(zeros(Nx, Ny, Nz, Nvc))
    for coil in 1:Nvc
        smaps_interp[:, :, :, coil] = complex.(
            imresize(real(smaps_crop[:, :, :, coil]), (Nx, Ny, Nz)),
            imresize(imag(smaps_crop[:, :, :, coil]), (Nx, Ny, Nz)),
        )
    end
    smaps_cpu = ComplexF32.(
        smaps_interp ./ (sqrt.(sum(abs2.(smaps_interp); dims=4)) .+ eps())
    )
    println("  Sensitivity maps: ", size(smaps_cpu))


    # ── 2. Load k-space ───────────────────────────────────────────────────────
    println("Loading k-space …")
    ksp0 = h5read(fn_ksp, "ksp_epi_zf")
    ksp0 = ComplexF32.([complex(k.real, k.imag) for k in ksp0])
    @assert size(ksp0) == (Nx, Ny, Nz, Nvc, Nt) "Unexpected k-space shape: $(size(ksp0))"


    # ── 3. Normalise k-space ──────────────────────────────────────────────────
    println("Normalising k-space …")
    img0 = sense_comb(ksp0, smaps_cpu)
    scale_factor = quantile(vec(abs.(img0)), 0.99)
    ksp0 ./= max(scale_factor, eps(Float32))
    println("  Scale factor = ", round(scale_factor; digits=4))


    # ── 4. Sampling mask ──────────────────────────────────────────────────────
    Ω = (ksp0[:, :, :, 1, :] .!= 0)
    R = prod(N) / sum(Ω[:, :, :, 1])
    println("Acceleration factor R ≈ ", round(R; digits=2))

    for ic in 2:Nvc
        @assert Ω == (ksp0[:, :, :, ic, :] .!= 0) "Coil $ic has a different sampling pattern"
    end
    for it in 2:Nt
        @assert sum(Ω[:, :, :, it]) == sum(Ω[:, :, :, it-1]) "Frame $it has different sample count"
    end


    # ── 5. Mask k-space ON CPU, then move only the sampled subset to GPU ──────
    #
    # CRITICAL: never call cu(ksp0).
    # ksp0 for N=(90,90,60), Nvc=18, Nt=387 is 27 GB — far exceeds 11 GB VRAM.
    # The masked ksp is ksp0/R ≈ 4.5 GB at R=6, which fits on GPU 0 for
    # single-GPU mode, but NOT alongside x_sum_buf and g_buf in multi-GPU mode.
    # In multi-GPU mode ksp stays on CPU and is consumed during Atksp computation.
    println("Masking k-space on CPU …")
    ksp0_flat = reshape(ksp0, Nx * Ny * Nz, Nvc, Nt)   # (Nvox, Nvc, Nt)
    ksp_cpu = cat([ksp0_flat[vec(s), :, it]
                   for (it, s) in enumerate(eachslice(Ω; dims=4))]...; dims=3)
    println("  Masked k-space: ", size(ksp_cpu),
        "  (", round(sizeof(ksp_cpu) / 1e9; digits=2), " GB on CPU)")

    ksp0 = nothing
    GC.gc()   # free 27 GB immediately


    # ── 6. Move smaps (and ksp for single-GPU) to GPU 0 ──────────────────────
    if use_gpu
        CUDA.device!(0)
        smaps = cu(smaps_cpu)
        if use_multigpu
            # ksp stays on CPU — consumed frame-by-frame to build Atksp below.
            # Moving 4.5 GB ksp to GPU 0 would leave only ~2 GB headroom for
            # x_sum_buf (1.5 GB) + g_buf (1.5 GB), causing OOM on 11 GB cards.
            println("Moving smaps to GPU 0 (ksp stays on CPU for frame-by-frame Atksp) …")
            ksp = ksp_cpu   # alias; freed after Atksp computation below
        else
            println("Moving smaps and masked k-space to GPU 0 …")
            ksp = cu(ksp_cpu)
            ksp_cpu = nothing
            GC.gc()
            GC.gc(true)
        end
        println("  GPU 0 free after transfer: ",
            round(CUDA.available_memory() / 1e9; digits=1), " GB")
    else
        smaps = smaps_cpu
        ksp = ksp_cpu
        ksp_cpu = nothing
    end


    # ── 7. SENSE encoding operator A ──────────────────────────────────────────
    println("Building encoding operator …")
    if use_gpu
        Aframe = (Ω_t, S) -> Asense_gpu(Ω_t, S; fft_forward=true, unitary=true)
    else
        Aframe = (Ω_t, S) -> Asense(Ω_t, S; fft_forward=true, unitary=true)
    end
    # Keep the per-frame list for multi-GPU frame-by-frame gradient;
    # block_diag is still used for single-GPU / CPU paths and power iteration.
    A_frames_list = [Aframe(s, smaps) for s in eachslice(Ω; dims=ndims(Ω))]
    A = block_diag(A_frames_list...)
    println("  Encoding operator built.")


    # ── 8. Lipschitz constant ─────────────────────────────────────────────────
    if isnothing(σ1A_PRECOMPUTED)
        println("Computing σ₁(A) via power iteration (may take ~20 min) …")
        _, σ1A = poweriter_mod(undim(A))
        println("  σ₁(A) = ", round(σ1A; digits=4))
    else
        σ1A = σ1A_PRECOMPUTED
    end
    L = Nscales * σ1A^2


    # ── 9. Regularisation weights ─────────────────────────────────────────────
    N_voxels = prod(N)
    λs = [
        sqrt(prod(PATCH_SIZES[k])) +
        sqrt(Nt) +
        sqrt(log(N_voxels * Nt / max(prod(PATCH_SIZES[k]), Nt)))
        for k in 1:Nscales
    ]
    println("Regularisation weights λs = ", round.(λs; digits=3))


    # ── 10. Reconstruct ───────────────────────────────────────────────────────

    if use_multigpu
        # ── Multi-GPU path ────────────────────────────────────────────────────

        # Flush CUDA memory pool on all devices before measuring free VRAM.
        # Without this, previous failed runs leave allocations in the pool
        # that show up as "used" even though they're reclaimable.
        for d in gpu_ids
            CUDA.device!(d)
            CUDA.reclaim()
        end
        CUDA.device!(0)

        # GPU 0 is the master (holds smaps, Atksp, SENSE frames).
        # Scale slices are distributed only among worker GPUs (gpu_ids[2:end]).
        worker_ids = gpu_ids[2:end]
        isempty(worker_ids) && error("Multi-GPU requires at least 2 GPUs.")

        # Measure free VRAM NOW — after smaps are on GPU 0 — on worker
        # devices only (GPU 0's headroom is irrelevant for scale allocation).
        img_sz = (Nx, Ny, Nz, Nt)
        min_free_gb = minimum(worker_ids) do d
            CUDA.device!(d)
            round(CUDA.available_memory() / 1e9; digits=1)
        end
        CUDA.device!(0)

        mom_type = recommend_mom(img_sz, Nscales, length(worker_ids), min_free_gb)
        peak_gb, n_live = estimate_vram(img_sz, Nscales, length(worker_ids);
            mom=mom_type)
        println("  Worker GPUs              : $(worker_ids)")
        println("  Tightest worker VRAM     : $(min_free_gb) GB")
        println("  Momentum selected        : $mom_type " *
                "($n_live bufs × $(round(prod(img_sz)*8/1e9;digits=1)) GB" *
                " = $(peak_gb) GB peak per worker)")

        # Pre-compute A'*ksp on GPU 0 frame-by-frame.
        # Peak transient: one frame of ksp (K×Nvc, ~12 MB) at a time.
        # Result Atksp has shape (Nx,Ny,Nz,Nt) = 1.5 GB — far less than
        # holding all of ksp (4.5 GB) on GPU 0.
        println("\nPre-computing A'*ksp on GPU 0 (frame-by-frame) …")
        CUDA.device!(0)
        A_frames_gpu = A_frames_list
        Atksp = CUDA.zeros(ComplexF32, Nx, Ny, Nz, Nt)
        norm_ksp_sq = Float64(norm(ksp))^2   # CPU norm, computed before freeing ksp
        for t in 1:Nt
            ksp_t = cu(ksp[:, :, t])   # (K, Nvc) → GPU, ~12 MB transient
            Atksp[:, :, :, t] .= reshape(A_frames_gpu[t]' * ksp_t, Nx, Ny, Nz)
        end
        ksp = nothing
        GC.gc()
        println("  GPU 0 free after Atksp: ",
            round(CUDA.available_memory() / 1e9; digits=1), " GB")

        println("\nAllocating multi-GPU buffers …")
        X0_cpu = zeros(ComplexF32, Nx, Ny, Nz, Nt, Nscales)
        X0_cpu[:, :, :, :, 1] = Array(Atksp)   # initialise first component from A'ksp
        mg = alloc_mg_bufs(X0_cpu, worker_ids, mom_type)
        X0_cpu = nothing
        GC.gc()

        # GPU-native cost function using the identity:
        #   (1/2)||A*x - ksp||² = (1/2)(||A*x||² - 2·Re(<x, A'ksp>) + ||ksp||²)
        # Avoids materialising a 4.5 GB residual buffer entirely.
        function Fcost_gpu(x_sum_gpu::CuArray)
            CUDA.device!(0)
            norm_Ax_sq = 0.0
            for t in 1:Nt
                Ax_t = A_frames_gpu[t] * vec(view(x_sum_gpu, :, :, :, t))
                norm_Ax_sq += real(Float64(dot(Ax_t, Ax_t)))
            end
            xdot = 2 * real(Float64(dot(vec(x_sum_gpu), vec(Atksp))))
            return 0.5 * (norm_Ax_sq - xdot + norm_ksp_sq)
        end

        reg_cost_mg(mg_state) = sum(
            λs[k] * patch_nucnorm(img2patches(Array(mg_state.xold[k]),
                PATCH_SIZES[k], STRIDES[k]))
            for k in 1:Nscales)

        # logger receives CPU arrays from pogm_mg; converts to GPU for cost call
        function logger_mg(iter, xk_sum_cpu, yk_sum_cpu, is_restart)
            CUDA.device!(0)
            xk_gpu = cu(xk_sum_cpu)
            dc = Fcost_gpu(xk_gpu)
            xk_gpu = nothing
            rc = reg_cost_mg(mg)
            return (dc, rc, is_restart)
        end

        println("\nRunning multi-GPU POGM " *
                "($NITERS iters, $Nscales scale(s), $mom_type, " *
                "$(length(gpu_ids)) GPU(s)) …")

        X_recon_cpu, costs = pogm_mg(
            mg, A_frames_gpu, Atksp,
            Fcost_gpu, λs, PATCH_SIZES, STRIDES, L, NITERS;
            mom=mom_type,
            restart=:gr,
            fun=logger_mg,
        )

        # Gather all scale components for saving
        X_cpu = cat([Array(mg.xnew[k]) for k in 1:Nscales]...; dims=5)

    elseif use_gpu
        # ── Single-GPU path (original logic) ─────────────────────────────────
        image_sum_sg(X) = dropdims(sum(X; dims=5); dims=5)

        dc_cost(X) = 0.5 * norm(A * image_sum_sg(X) - ksp)^2
        dc_cost_grad(X) = repeat(A' * (A * image_sum_sg(X) - ksp);
            outer=[1, 1, 1, 1, Nscales])
        reg_cost(X) = sum(
            λs[k] * patch_nucnorm(img2patches(view(X, :, :, :, :, k),
                PATCH_SIZES[k], STRIDES[k]))
            for k in 1:Nscales)

        g_prox = (X, c) -> begin
            for k in 1:Nscales
                @views X[:, :, :, :, k] = patchSVST_gpu(
                    X[:, :, :, :, k], c * λs[k], PATCH_SIZES[k], STRIDES[k])
            end
            X
        end

        comp_cost = X -> dc_cost(X) + reg_cost(X)
        logger = (iter, xk, yk, rs) -> (dc_cost(xk), reg_cost(xk), rs)

        X0 = CUDA.zeros(ComplexF32, Nx, Ny, Nz, Nt, Nscales)
        X0[:, :, :, :, 1] = A' * ksp

        println("\nRunning POGM ($NITERS iters, $Nscales scale(s), single GPU) …")
        X, costs = pogm_mod(X0, comp_cost, dc_cost_grad, L;
            mom=:pogm, niter=NITERS, g_prox=g_prox, fun=logger)

        X_cpu = Array(X)
        X_recon_cpu = dropdims(sum(X_cpu; dims=5); dims=5)

    else
        # ── CPU path (original logic) ─────────────────────────────────────────
        image_sum_cpu(X) = dropdims(sum(X; dims=5); dims=5)

        dc_cost(X) = 0.5 * norm(A * image_sum_cpu(X) - ksp)^2
        dc_cost_grad(X) = repeat(A' * (A * image_sum_cpu(X) - ksp);
            outer=[1, 1, 1, 1, Nscales])
        reg_cost(X) = sum(
            λs[k] * patch_nucnorm(img2patches(view(X, :, :, :, :, k),
                PATCH_SIZES[k], STRIDES[k]))
            for k in 1:Nscales)

        g_prox = (X, c) -> begin
            for k in 1:Nscales
                @views X[:, :, :, :, k] = patchSVST(
                    X[:, :, :, :, k], c * λs[k], PATCH_SIZES[k], STRIDES[k])
            end
            X
        end

        comp_cost = X -> dc_cost(X) + reg_cost(X)
        logger = (iter, xk, yk, rs) -> (dc_cost(xk), reg_cost(xk), rs)

        X0 = zeros(ComplexF32, Nx, Ny, Nz, Nt, Nscales)
        X0[:, :, :, :, 1] = A' * ksp

        println("\nRunning POGM ($NITERS iters, $Nscales scale(s), CPU) …")
        X, costs = pogm_mod(X0, comp_cost, dc_cost_grad, L;
            mom=:pogm, niter=NITERS, g_prox=g_prox, fun=logger)

        X_cpu = X
        X_recon_cpu = dropdims(sum(X_cpu; dims=5); dims=5)
    end


    # ── 11. Extract cost logs ─────────────────────────────────────────────────
    dc_costs = [c[1] for c in costs]
    reg_costs = [c[2] for c in costs]
    restarts = [c[3] for c in costs]


    # ── 12. Save ──────────────────────────────────────────────────────────────
    fn_out = fn_recon_base * "_$(Nscales)scales.mat"
    matwrite(fn_out, Dict(
            "X" => X_cpu,
            "X_recon" => X_recon_cpu,
            "omega" => Ω,
            "dc_costs" => dc_costs,
            "reg_costs" => reg_costs,
            "restarts" => restarts,
            "R" => R,
            "sigma1A" => σ1A,
            "L" => L,
            "Nscales" => Nscales,
            "patch_sizes" => PATCH_SIZES,
            "strides" => STRIDES,
            "lambdas" => λs,
            "Niters" => NITERS,
            "scale_factor" => scale_factor,
            "used_gpu" => use_gpu,
            "num_gpus" => use_multigpu ? length(gpu_ids) : (use_gpu ? 1 : 0),
            "mom_type" => string(mom_type),
        ); compress=true)

    println("\n✓ Saved → $fn_out")
end

end # module Reconstruct