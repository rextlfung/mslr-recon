module MultiGPU

export detect_gpus, assign_scales, estimate_vram, recommend_mom,
    alloc_mg_bufs, pogm_mg, image_sum, image_sum_into!,
    broadcast_to_fgrad!, apply_prox!, MGState

#=
multigpu.jl
Multi-GPU orchestration for MSLR fMRI reconstruction.

Algorithm
─────────
MSLR POGM minimises:
  F(X) = (1/2)||A(Σ_k X_k) - ksp||² + Σ_k λ_k ||P_k(X_k)||_*

∇f(X)_k = A'*(A*(Σ_k X_k) - ksp)   for all k  (same gradient, repeated)
prox_k(X_k, c) = patchSVST(X_k, c*λ_k)         for each k independently

The 5D POGM iterate X has Nscales independent components, each living on
its assigned GPU.  All POGM arithmetic (momentum, znew, etc.) involves
linear combinations of X slices with SCALAR coefficients — so each device
can update its own slice(s) identically, with only the scalar coefficients
communicated (these are Float64 scalars, essentially zero communication cost).

The ONLY cross-device communication per iteration is:
  1. image_sum: Σ_k X_k  →  each device sends its slice to GPU 0  (~1.4 GB/slice)
  2. gradient scalar g = A'*(A*x_sum - ksp) computed on GPU 0  (~1.4 GB)
  3. g broadcast from GPU 0 to each device  (~1.4 GB/device)

Everything else (znew, unew, Fgrad, ynew, momentum) is computed LOCALLY on
each device using only that device's own slice and the shared scalar coefficients.

Buffer layout per device (per scale k on device dev_of[k]):
  X_k  slice     : (Nx,Ny,Nz,Nt) ComplexF32  — the iterate
  y_k, u_k, z_k  : same shape  — POGM state (yold, uold, zold)
  xn_k, yn_k     : same shape  — xnew, ynew
  un_k, zn_k     : same shape  — unew, znew
  fg_k, Fg_k     : same shape  — fgrad (gradient copy), Fgrad
  Fg0_k          : same shape  — Fgradold
  Total: 11 buffers per scale × 1.4 GB = 15.4 GB per scale at Nt=387.

At 11 GB per device this requires Nt ≤ ~280 for 1 scale/device, or
use fewer buffers by choosing mom=:fpgm (7 buffers = 9.8 GB, fits for Nt=387).

mom buffer counts:
  :pogm  11 buffers × 1.4 GB ≈ 15.4 GB  (needs 24 GB GPU for Nt=387)
  :fpgm   7 buffers × 1.4 GB ≈  9.8 GB  (fits RTX 2080 Ti for Nt=387) ✓
  :pgm    5 buffers × 1.4 GB ≈  7.0 GB  (fits comfortably)

reconstruct.jl auto-selects mom via recommend_mom() based on free VRAM.

Rex Fung, University of Michigan
=#

using CUDA
using LinearAlgebra
using ProgressMeter
using Printf


# ─────────────────────────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    detect_gpus() -> Vector{Int}

Return 0-based CUDA device indices of all functional GPUs.
"""
function detect_gpus()::Vector{Int}
    devs = collect(CUDA.devices())
    isempty(devs) && error("No CUDA devices found.")
    println("Detected $(length(devs)) CUDA device(s):")
    functional = Int[]
    for dev in devs
        idx = CUDA.deviceid(dev)
        try
            CUDA.device!(dev)
            free_gb = round(CUDA.available_memory() / 1e9; digits=1)
            total_gb = round(CUDA.total_memory() / 1e9; digits=1)
            @printf("  [%d] %-36s  %5.1f / %5.1f GB free\n",
                idx, CUDA.name(dev), free_gb, total_gb)
            push!(functional, idx)
        catch e
            @warn "Device $idx skipped: $e"
        end
    end
    CUDA.device!(0)
    return functional
end


# ─────────────────────────────────────────────────────────────────────────────
# Memory estimation and momentum selection
# ─────────────────────────────────────────────────────────────────────────────

_nbuf(mom::Symbol) = Dict(:pogm => 11, :fpgm => 7, :pgm => 5)[mom]

"""
    estimate_vram(img_sz, Nscales, n_workers; mom) -> (peak_gb_per_device, n_buffers)

Peak VRAM per worker device (GPU 0 excluded — it only holds smaps/ksp/SENSE).
`n_workers` = length(gpu_ids) - 1.
"""
function estimate_vram(img_sz::NTuple{4,Int}, Nscales::Int, n_workers::Int;
    mom::Symbol=:pogm)
    n = _nbuf(mom)
    spd = cld(Nscales, n_workers)        # max scales per worker device
    bytes = prod(img_sz) * 8 * n * spd     # ComplexF32 = 8 bytes
    return round(bytes / 1e9; digits=1), n
end

"""
    recommend_mom(img_sz, Nscales, n_workers, free_vram_gb) -> Symbol

Choose the most aggressive momentum type that fits on the worker devices
(GPU 0 excluded) with 15% headroom.
"""
function recommend_mom(img_sz::NTuple{4,Int}, Nscales::Int, n_workers::Int,
    free_vram_gb::Float64)::Symbol
    for mom in (:pogm, :fpgm, :pgm)
        peak, _ = estimate_vram(img_sz, Nscales, n_workers; mom)
        peak < 0.85 * free_vram_gb && return mom
    end
    peak_pgm, _ = estimate_vram(img_sz, Nscales, n_workers; mom=:pgm)
    error("""
    Insufficient VRAM for multi-GPU reconstruction.
    :pgm (minimum) needs $(peak_pgm) GB per worker device; available: $(free_vram_gb) GB.
    Options: reduce Nt (currently $(img_sz[4])), reduce Nscales ($Nscales), or use GPUs with more VRAM.
    """)
end


# ─────────────────────────────────────────────────────────────────────────────
# Scale → device assignment
# ─────────────────────────────────────────────────────────────────────────────

"""
    assign_scales(Nscales, worker_ids) -> Vector{Int}

Round-robin assignment of scales to worker GPUs (GPU 0 excluded).
Returns dev_of[k] = CUDA device index for scale k.
"""
function assign_scales(Nscales::Int, worker_ids::Vector{Int})::Vector{Int}
    n = length(worker_ids)
    dv = [worker_ids[mod1(k, n)] for k in 1:Nscales]
    println("Scale → GPU assignment (GPU 0 reserved for encoding operator):")
    for k in 1:Nscales
        println("  scale $k → GPU $(dv[k])")
    end
    return dv
end


# ─────────────────────────────────────────────────────────────────────────────
# MGState — per-scale buffer pool
# ─────────────────────────────────────────────────────────────────────────────

"""
    MGState

Holds all per-scale CuArray buffers.  Each buffer vector has length Nscales;
element k lives on device dev_of[k].

Fields named to match the POGM variables in mirt_mod.jl:
  xold, yold, uold, zold   — state from previous iteration
  xnew, ynew               — new iterates (pre-allocated)
  unew, znew               — auxiliary (pre-allocated)
  fgrad                    — ∇f copy per scale
  Fgrad, Fgradold          — gradient for restart check
"""
struct MGState
    xold::Vector{CuArray{ComplexF32,4}}
    yold::Vector{CuArray{ComplexF32,4}}
    uold::Vector{CuArray{ComplexF32,4}}
    zold::Vector{CuArray{ComplexF32,4}}
    xnew::Vector{CuArray{ComplexF32,4}}
    ynew::Vector{CuArray{ComplexF32,4}}
    unew::Vector{CuArray{ComplexF32,4}}
    znew::Vector{CuArray{ComplexF32,4}}
    fgrad::Vector{CuArray{ComplexF32,4}}
    Fgrad::Vector{CuArray{ComplexF32,4}}
    Fgradold::Vector{CuArray{ComplexF32,4}}
    dev_of::Vector{Int}
    img_sz::NTuple{4,Int}
    Nscales::Int
end

"""
    alloc_mg_bufs(X0_cpu, worker_ids, mom) -> MGState

Allocate POGM buffers for each scale on its assigned worker device.
`worker_ids` must NOT include GPU 0 (the master device).
Only allocates the buffers actually needed for `mom`:
  :pogm → 11 buffers  :fpgm → 7  :pgm → 5
"""
function alloc_mg_bufs(X0_cpu::Array{ComplexF32,5},
    worker_ids::Vector{Int},
    mom::Symbol=:pogm)::MGState
    0 ∈ worker_ids && error("GPU 0 is the master device and must not be in worker_ids. " *
                            "Pass gpu_ids[2:end] (got $worker_ids).")
    Ns = size(X0_cpu, 5)
    img_sz = size(X0_cpu)[1:4]
    dev_of = assign_scales(Ns, worker_ids)

    function _alloc_vec(init_fn)
        [
            begin
                CUDA.device!(dev_of[k])
                init_fn(k)
            end for k in 1:Ns
        ]
    end

    _z(k) = CUDA.zeros(ComplexF32, img_sz)
    _x(k) = CuArray(X0_cpu[:, :, :, :, k])

    # Allocate only the buffers needed for the chosen momentum type.
    # :pogm needs all 11; :fpgm skips uold/zold/unew/znew (7 total);
    # :pgm additionally skips Fgradold (5 total).
    # Unused buffers are filled with empty CuArrays to keep MGState concrete.
    _empty(k) = CUDA.zeros(ComplexF32, ntuple(_ -> 0, 4))  # zero-size placeholder

    need_uz = mom === :pogm                  # uold, zold, unew, znew
    need_Fg0 = mom !== :pgm                   # Fgradold

    xold = _alloc_vec(_x)
    yold = _alloc_vec(_x)
    uold = need_uz ? _alloc_vec(_x) : _alloc_vec(_empty)
    zold = need_uz ? _alloc_vec(_x) : _alloc_vec(_empty)
    xnew = _alloc_vec(_z)
    ynew = _alloc_vec(_z)
    unew = need_uz ? _alloc_vec(_z) : _alloc_vec(_empty)
    znew = need_uz ? _alloc_vec(_z) : _alloc_vec(_empty)
    fgrad = _alloc_vec(_z)
    Fgrad = _alloc_vec(_z)
    Fgradold = need_Fg0 ? _alloc_vec(_z) : _alloc_vec(_empty)

    CUDA.device!(0)
    return MGState(xold, yold, uold, zold, xnew, ynew, unew, znew,
        fgrad, Fgrad, Fgradold, dev_of, img_sz, Ns)
end


# ─────────────────────────────────────────────────────────────────────────────
# image_sum — async gather of all scale slices to GPU 0
# ─────────────────────────────────────────────────────────────────────────────

"""
    image_sum(mg, bufs_field) -> CuArray{ComplexF32,4} on GPU 0

Sum a named buffer across all scales onto GPU 0.
`bufs_field` is a Vector{CuArray} from MGState, e.g. `mg.xold`.
"""
function image_sum(mg::MGState, field::Vector{<:CuArray})::CuArray{ComplexF32,4}
    CUDA.device!(0)
    total = CUDA.zeros(ComplexF32, mg.img_sz)
    tasks = [Threads.@spawn begin
        if mg.dev_of[k] == 0
            copy(field[k])
        else
            dst = CuArray{ComplexF32}(undef, mg.img_sz)
            copyto!(dst, field[k])
            dst
        end
    end for k in 1:mg.Nscales]
    for t in tasks
        CUDA.device!(0)
        total .+= fetch(t)
    end
    return total
end

"""
    image_sum_into!(buf, mg, field=mg.xold)

In-place version of image_sum: gathers all scale slices and sums them
into the pre-allocated `buf` on GPU 0, avoiding a fresh allocation.
"""
function image_sum_into!(buf::CuArray, mg::MGState,
    field::Vector{<:CuArray}=mg.xold)
    CUDA.device!(0)
    fill!(buf, zero(ComplexF32))
    tasks = [Threads.@spawn begin
        if mg.dev_of[k] == 0
            copy(field[k])
        else
            dst = CuArray{ComplexF32}(undef, mg.img_sz)
            copyto!(dst, field[k])
            dst
        end
    end for k in 1:mg.Nscales]
    for t in tasks
        CUDA.device!(0)
        buf .+= fetch(t)
    end
    return buf
end

# Convenience: sum xold (the current iterate)
image_sum_x(mg::MGState) = image_sum(mg, mg.xold)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient broadcast
# ─────────────────────────────────────────────────────────────────────────────

"""
    broadcast_to_fgrad!(mg, g_gpu0)

Copy a (Nx,Ny,Nz,Nt) CuArray from GPU 0 into each scale's fgrad buffer
on its assigned device.  Async across devices.
"""
function broadcast_to_fgrad!(mg::MGState, g_gpu0::CuArray)
    tasks = [Threads.@spawn begin
        if mg.dev_of[k] == 0
            copyto!(mg.fgrad[k], g_gpu0)
        else
            CUDA.device!(mg.dev_of[k])
            copyto!(mg.fgrad[k], g_gpu0)   # peer copy
        end
    end for k in 1:mg.Nscales]
    foreach(fetch, tasks)
    CUDA.device!(0)
end


# ─────────────────────────────────────────────────────────────────────────────
# Proximal operator
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_prox!(mg, src_field, dst_field, c, λs, PATCH_SIZES, STRIDES)

In-place patchSVST on each device in parallel.
  dst_field[k] = patchSVST(src_field[k], c * λs[k])
"""
function apply_prox!(mg::MGState,
    src::Vector{<:CuArray}, dst::Vector{<:CuArray},
    c::Real, λs, PATCH_SIZES, STRIDES)
    tasks = [Threads.@spawn begin
        CUDA.device!(mg.dev_of[k])
        dst[k] .= Main.SenseGPU.patchSVST_gpu(
            src[k], Float32(c * λs[k]),
            PATCH_SIZES[k], STRIDES[k])
    end for k in 1:mg.Nscales]
    foreach(fetch, tasks)
    CUDA.device!(0)
end


# ─────────────────────────────────────────────────────────────────────────────
# Per-device in-place arithmetic helpers
# ─────────────────────────────────────────────────────────────────────────────

# dst[k] .= a*src1[k] .+ b*src2[k]   for all k in parallel
function _axpby_field!(mg, dst, src1, src2, a::Real, b::Real)
    tasks = [Threads.@spawn begin
        CUDA.device!(mg.dev_of[k])
        dst[k] .= a .* src1[k] .+ b .* src2[k]
    end for k in 1:mg.Nscales]
    foreach(fetch, tasks)
    CUDA.device!(0)
end

# dst[k] .= A*s1[k] + B*s2[k] + C*s3[k] + D*s4[k]  (fused z update)
function _fused_z_field!(mg, dst, s1, s2, s3, s4,
    A::Real, B::Real, C::Real, D::Real)
    tasks = [Threads.@spawn begin
        CUDA.device!(mg.dev_of[k])
        dst[k] .= A .* s1[k] .+ B .* s2[k] .+
                  C .* s3[k] .+ D .* s4[k]
    end for k in 1:mg.Nscales]
    foreach(fetch, tasks)
    CUDA.device!(0)
end

# Parallel copyto! across all scales
function _copy_field!(mg, dst, src)
    tasks = [Threads.@spawn begin
        CUDA.device!(mg.dev_of[k])
        copyto!(dst[k], src[k])
    end for k in 1:mg.Nscales]
    foreach(fetch, tasks)
    CUDA.device!(0)
end

# Parallel norm² sum across all scale buffers on their devices
function _norm2_field(mg, field)
    tasks = [Threads.@spawn begin
        CUDA.device!(mg.dev_of[k])
        Float64(CUDA.norm(field[k]))^2
    end for k in 1:mg.Nscales]
    return sum(fetch(t) for t in tasks)
end

# sum(real(a[k] .* b[k])) across all k — for restart dot check
function _rdot_field(mg, a, b)
    tasks = [Threads.@spawn begin
        CUDA.device!(mg.dev_of[k])
        real(Float64(dot(vec(a[k]), vec(b[k]))))
    end for k in 1:mg.Nscales]
    return sum(fetch(t) for t in tasks)
end


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU POGM
# ─────────────────────────────────────────────────────────────────────────────

"""
    x_sum, out = pogm_mg(mg, A, ksp, Fcost, λs, PATCH_SIZES, STRIDES, L, NITERS;
                          mom, restart, restart_cutoff, bsig, f_mu, fun)

Full multi-GPU POGM for MSLR.  All POGM buffers live in `mg` (an MGState);
no heap allocation occurs inside the loop.

`A` and `ksp` live on GPU 0.  The gradient A'*(A*Σ X_k - ksp) is computed
on GPU 0, then broadcast to each device's fgrad buffer.  Per-device arithmetic
(znew, unew, etc.) is dispatched in parallel via Threads.@spawn.

Returns the summed image (CPU Array) and the cost log vector.
"""
function pogm_mg(
    mg::MGState,
    A,
    ksp::CuArray,
    Fcost::Function,
    λs::Vector{<:Real},
    PATCH_SIZES::Vector,
    STRIDES::Vector,
    L::Real,
    NITERS::Int;
    mom::Symbol=:pogm,
    restart::Symbol=:gr,
    restart_cutoff::Real=0.,
    bsig::Real=1.,
    f_mu::Real=0.,
    fun::Function=(iter, xk_sum, yk_sum, rs) -> undef,
)
    mom ∈ (:pgm, :fpgm, :pogm) || throw(ArgumentError("mom=$mom"))
    restart ∈ (:none, :gr, :fr) || throw(ArgumentError("restart=$restart"))

    alpha = 1.0 / Float64(L)
    mu = Float64(f_mu)
    q = mu / Float64(L)

    # Pre-allocate fixed-size buffers on GPU 0 for the gradient step.
    # These are reused every iteration — no transient 4.5 GB allocations.
    CUDA.device!(0)
    x_sum_buf = CUDA.zeros(ComplexF32, mg.img_sz)   # Σ X_k
    residual_buf = similar(ksp)                         # A*x - ksp  (K,Nvc,Nt)
    g_buf = CUDA.zeros(ComplexF32, mg.img_sz)   # A'*residual

    told = 1.0
    sig = 1.0
    zetaold = 1.0
    Fcostold = Fcost(Array(image_sum_x(mg)))

    out = Vector{Any}(undef, NITERS + 1)
    out[1] = fun(0, Array(image_sum_x(mg)), Array(image_sum_x(mg)), false)

    @showprogress 1 "Multi-GPU POGM ($mom)..." for iter in 1:NITERS

        is_restart = false

        # ── 1. Gradient on GPU 0 (fully in-place, no transient allocations) ──
        CUDA.device!(0)
        image_sum_into!(x_sum_buf, mg)                # gather Σ X_k → x_sum_buf
        residual_buf .= A * x_sum_buf .- ksp          # reuse residual_buf in-place
        g_buf .= A' * residual_buf                    # gradient → g_buf in-place
        broadcast_to_fgrad!(mg, g_buf)                # push to each worker device

        # ── 2. Momentum update (per-device, in parallel) ──────────────────────
        if mom === :pgm || mom === :fpgm

            # unew[k] = xold[k] - alpha * fgrad[k]
            _axpby_field!(mg, mg.unew, mg.xold, mg.fgrad, 1.0, -alpha)

            # ynew[k] = prox(unew[k], alpha * lambda_k)
            apply_prox!(mg, mg.unew, mg.ynew, alpha, λs, PATCH_SIZES, STRIDES)

            # Fgrad[k] = (1/alpha) * (xold[k] - ynew[k])
            _axpby_field!(mg, mg.Fgrad, mg.xold, mg.ynew, 1 / alpha, -1 / alpha)

            Fcostnew = Fcost(Array(image_sum(mg, mg.ynew)))

            if restart !== :none
                if restart === :fr && Fcostnew > Fcostold
                    told = 1.0
                    is_restart = true
                elseif restart === :gr
                    # ynew - yold into znew[k] as scratch
                    _axpby_field!(mg, mg.znew, mg.ynew, mg.yold, 1.0, -1.0)
                    lhs = -_rdot_field(mg, mg.Fgrad, mg.znew)
                    rhs = restart_cutoff * sqrt(_norm2_field(mg, mg.Fgrad)) *
                          sqrt(_norm2_field(mg, mg.znew))
                    lhs <= rhs && (told = 1.0; is_restart = true)
                end
                Fcostold = Fcostnew
            end

            if mom === :fpgm
                tnew = (mu != 0) ? told : 0.5 * (1 + sqrt(1 + 4 * told^2))
                beta = (mu != 0) ? (1 - sqrt(q)) / (1 + sqrt(q)) : (told - 1) / tnew
                # xnew[k] = (1+beta)*ynew[k] - beta*yold[k]
                _axpby_field!(mg, mg.xnew, mg.ynew, mg.yold, 1 + beta, -beta)
            else  # :pgm
                _copy_field!(mg, mg.xnew, mg.ynew)
                tnew = told
            end

        else  # :pogm ───────────────────────────────────────────────────────────

            tnew = (iter == NITERS) ?
                   0.5 * (1 + sqrt(1 + 8 * told^2)) :
                   0.5 * (1 + sqrt(1 + 4 * told^2))
            beta = (mu != 0) ? (2 + q - sqrt(q^2 + 8q))^2 / 4 / (1 - q) : (told - 1) / tnew
            gamma = (mu != 0) ? (2 + q - sqrt(q^2 + 8q)) / 2 : sig * told / tnew

            # unew[k] = xold[k] - alpha*fgrad[k]
            _axpby_field!(mg, mg.unew, mg.xold, mg.fgrad, 1.0, -alpha)

            # znew[k] = (1+β+γ)*unew - β*uold - (γ+βα/ζ)*xold + (βα/ζ)*zold
            coeff_d = beta * alpha / zetaold
            _fused_z_field!(mg, mg.znew, mg.unew, mg.uold, mg.xold, mg.zold,
                1 + beta + gamma, -beta, -(gamma + coeff_d), coeff_d)

            zetanew = alpha * (1 + beta + gamma)

            # xnew[k] = prox(znew[k], zetanew * lambda_k)
            apply_prox!(mg, mg.znew, mg.xnew, zetanew, λs, PATCH_SIZES, STRIDES)

            # Fgrad[k] = fgrad[k] - (1/zetanew)*(xnew[k] - znew[k])
            tasks = [Threads.@spawn begin
                CUDA.device!(mg.dev_of[k])
                mg.Fgrad[k] .= mg.fgrad[k] .-
                               (1 / zetanew) .* (mg.xnew[k] .- mg.znew[k])
            end for k in 1:mg.Nscales]
            foreach(fetch, tasks)
            CUDA.device!(0)

            # ynew[k] = xold[k] - alpha*Fgrad[k]
            _axpby_field!(mg, mg.ynew, mg.xold, mg.Fgrad, 1.0, -alpha)

            Fcostnew = Fcost(Array(image_sum(mg, mg.xnew)))

            if restart !== :none
                if restart === :fr && Fcostnew > Fcostold
                    tnew = 1.0
                    sig = 1.0
                    is_restart = true
                elseif restart === :gr
                    _axpby_field!(mg, mg.unew, mg.ynew, mg.yold, 1.0, -1.0) # scratch
                    lhs = -_rdot_field(mg, mg.Fgrad, mg.unew)
                    rhs = restart_cutoff * sqrt(_norm2_field(mg, mg.Fgrad)) *
                          sqrt(_norm2_field(mg, mg.unew))
                    lhs <= rhs && (tnew = 1.0; sig = 1.0; is_restart = true)
                end
                if !is_restart && _rdot_field(mg, mg.Fgrad, mg.Fgradold) < 0
                    sig = bsig * sig
                end
                Fcostold = Fcostnew
                _copy_field!(mg, mg.Fgradold, mg.Fgrad)
            end

            _copy_field!(mg, mg.uold, mg.unew)
            _copy_field!(mg, mg.zold, mg.znew)
            zetaold = zetanew
        end

        # Log (gather sums; only for logging, skipped if fun returns undef)
        xnew_sum_arr = Array(image_sum(mg, mg.xnew))
        ynew_sum_arr = Array(image_sum(mg, mg.ynew))
        out[iter+1] = fun(iter, xnew_sum_arr, ynew_sum_arr, is_restart)

        _copy_field!(mg, mg.xold, mg.xnew)
        _copy_field!(mg, mg.yold, mg.ynew)
        iszero(mu) && mom !== :pgm && (told = tnew)
    end

    final_field = (mom === :pogm) ? mg.xnew : mg.ynew
    return Array(image_sum(mg, final_field)), out
end

end # module MultiGPU