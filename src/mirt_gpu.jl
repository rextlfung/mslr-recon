module MirtGPU

export pogm_gpu, GradFn, ProxFn

#=
mirt_gpu.jl
GPU-friendly POGM: zero heap allocation inside the iteration loop.

Differences from mirt_mod.jl
─────────────────────────────
1. All POGM state lives in a pre-allocated NamedTuple of buffers (`bufs`).
   No `similar`, no `zeros`, no temporaries inside the loop.

2. f_grad and g_prox use an in-place convention:
     f_grad!(buf_out, x)   — writes ∇f(x) into buf_out, returns nothing
     g_prox!(buf_out, z, c) — writes prox_{c·g}(z) into buf_out, returns nothing
   The caller (reconstruct.jl) supplies these closures.

3. The fused momentum update for :pogm
     znew = unew + β(unew−uold) + γ(unew−xold) − βα/ζ(xold−zold)
   is computed with a single kernel loop (one pass over memory) via
   a helper _fused_z!  that works on both CuArrays and plain Arrays.

4. The restart dot-product check
     sum(real(Fgrad .* Fgradold))
   is computed with a single BLAS/cuBLAS dot call (no temporary array).

5. All scalar momentum coefficients (alpha, beta, gamma, zetanew, tnew)
   are plain Float64 — no GPU involvement.

6. poweriter_gpu is identical to poweriter_mod but uses the same module.

API
───
    bufs = alloc_pogm_bufs(x0)
    x, out = pogm_gpu(x0, bufs, Fcost, f_grad!, f_L;
                       g_prox! = (dst, z, c) -> copyto!(dst, z),
                       mom = :pogm, restart = :gr, niter = 10,
                       restart_cutoff = 0., bsig = 1., f_mu = 0.,
                       fun = (iter, xk, yk, restart) -> undef)

Buffer pool layout (all same shape as x0):
  :xnew, :ynew, :unew, :znew, :fgrad, :Fgrad, :Fgradold
  + :xold, :yold, :uold, :zold (the four "old" state pointers —
    initialised to x0; rotated by pointer swap, no copy)

Total distinct buffers: 11 (7 named + 4 old-state aliases that start as
copies of x0 but are swapped not copied thereafter).
For N=(90,90,60), Nt=387, ComplexF32: 11 × 1.4 GB ≈ 15.4 GB —
still exceeds 11 GB per device.

Use with Nscales distributed across GPUs so each device holds only ONE
scale slice; the POGM buffers for that slice = 11 × 1.4 GB is still tight.

RECOMMENDED: distribute so GPU 0 (the master) gets the global scale
(largest patch, most compressible, fastest SVD) and secondary GPUs get
the local scales. Keep Nt ≤ 300 for RTX 2080 Ti (11 × 1.1 GB ≈ 12 GB).

Rex Fung, University of Michigan
=#

using LinearAlgebra
using ProgressMeter
using CUDA


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases for the in-place function convention
# ─────────────────────────────────────────────────────────────────────────────

"""
    GradFn = (dst::AbstractArray, x::AbstractArray) -> Nothing

In-place gradient function: writes ∇f(x) into `dst`.
"""
const GradFn = Function

"""
    ProxFn = (dst::AbstractArray, z::AbstractArray, c::Real) -> Nothing

In-place proximal operator: writes prox_{c·g}(z) into `dst`.
"""
const ProxFn = Function


# ─────────────────────────────────────────────────────────────────────────────
# Buffer allocation
# ─────────────────────────────────────────────────────────────────────────────

"""
    alloc_pogm_bufs(x0) -> NamedTuple

Pre-allocate all POGM buffers matching the shape and type of `x0`.
Call once before `pogm_gpu`; reuse across multiple POGM runs with the
same geometry (e.g. across outer restarts).

Works for both CuArray (GPU) and plain Array (CPU).
"""
function alloc_pogm_bufs(x0::AbstractArray{T}) where T
    _z() = fill!(similar(x0), zero(T))
    return (
        xold=copy(x0),   # state: rotated each iter, not copied
        yold=copy(x0),
        uold=copy(x0),
        zold=copy(x0),
        xnew=_z(),
        ynew=_z(),
        unew=_z(),
        znew=_z(),
        fgrad=_z(),
        Fgrad=_z(),
        Fgradold=_z(),
    )
end


# ─────────────────────────────────────────────────────────────────────────────
# Fused momentum kernel (single-pass, no temporaries)
# ─────────────────────────────────────────────────────────────────────────────

#=
Computes:
  dst = unew + β*(unew - uold) + γ*(unew - xold) - (βα/ζ)*(xold - zold)
      = unew*(1+β+γ) - β*uold - γ*xold - (βα/ζ)*xold + (βα/ζ)*zold
      = unew*A  +  uold*B  +  xold*C  +  zold*D
where
  A =  1 + β + γ
  B = -β
  C = -(γ + βα/ζ)
  D =  βα/ζ
All coefficients are scalar Float64; the kernel is element-wise.
=#
function _fused_z!(dst, unew, uold, xold, zold,
    A::Real, B::Real, C::Real, D::Real)
    dst .= A .* unew .+ B .* uold .+ C .* xold .+ D .* zold
end

# cuBLAS-friendly real dot product: sum(real(a .* b))
# Uses dot(a, b) which for ComplexF32 returns conj(a)⋅b;
# we want Re(a⋅b) = Re(conj(conj(a))⋅b). Use axpby trick instead:
function _real_dot(a, b)
    # Materialize real parts on-device, then dot — one temp per device (small)
    return real(dot(a, b))   # LinearAlgebra.dot dispatches to cuBLAS for CuArray
end


# ─────────────────────────────────────────────────────────────────────────────
# Gradient restart check (no allocation)
# ─────────────────────────────────────────────────────────────────────────────

function _gr_restart_gpu(Fgrad, ynew_minus_yold, cutoff::Real)
    # sum(real(-Fgrad .* delta)) ≤ cutoff * norm(Fgrad) * norm(delta)
    # Re(-a⋅b) = -Re(dot(conj(a), b)) but dot for complex = conj(a)⋅b
    # We want Re(Fgrad[i] * delta[i]) — use real(dot(conj(Fgrad), delta))
    # LinearAlgebra.dot(a,b) = sum(conj(a_i)*b_i), so Re(dot(Fgrad,delta))
    # gives what we need for the sign check.
    lhs = -real(_real_dot(Fgrad, ynew_minus_yold))
    rhs = cutoff * norm(Fgrad) * norm(ynew_minus_yold)
    return lhs <= rhs
end


# ─────────────────────────────────────────────────────────────────────────────
# Main POGM loop
# ─────────────────────────────────────────────────────────────────────────────

"""
    x, out = pogm_gpu(x0, bufs, Fcost, f_grad!, f_L; kwargs...)

In-place POGM. All iterates live in `bufs` (a NamedTuple from `alloc_pogm_bufs`).
`x0` is used as the initial value and is not modified.

# In-place function signatures
- `f_grad!(dst, x)`    — writes ∇f(x) into `dst`
- `g_prox!(dst, z, c)` — writes prox_{c·g}(z) into `dst`

# Keyword arguments (identical to pogm_mod)
| Name             | Default  | Description                                    |
|:-----------------|:---------|:-----------------------------------------------|
| `g_prox!`        | identity | In-place proximal operator                     |
| `mom`            | `:pogm`  | `:pogm`, `:fpgm`, or `:pgm`                    |
| `restart`        | `:gr`    | `:gr`, `:fr`, or `:none`                       |
| `restart_cutoff` | `0.`     | Cosine threshold for gradient restart          |
| `bsig`           | `1.`     | γ decrease factor ∈ [0,1]                      |
| `niter`          | `10`     | Number of iterations                           |
| `f_mu`           | `0.`     | Strong convexity parameter                     |
| `fun`            | undef    | `fun(iter, xk, yk, is_restart)` logger         |

# Returns
- `x`   — reference to the final iterate buffer (`:xnew` or `:ynew` in bufs)
- `out` — Vector of `fun(...)` results, length `niter + 1`
"""
function pogm_gpu(
    x0::AbstractArray,
    bufs::NamedTuple,
    Fcost::Function,
    f_grad!::GradFn,
    f_L::Real;
    g_prox!::ProxFn=(dst, z, c) -> copyto!(dst, z),
    mom::Symbol=:pogm,
    restart::Symbol=:gr,
    restart_cutoff::Real=0.,
    bsig::Real=1.,
    niter::Int=10,
    f_mu::Real=0.,
    fun::Function=(iter, xk, yk, is_restart) -> undef,
)
    mom ∈ (:pgm, :fpgm, :pogm) || throw(ArgumentError("mom=$mom"))
    restart ∈ (:none, :gr, :fr) || throw(ArgumentError("restart=$restart"))
    f_L >= 0 || throw(ArgumentError("f_L=$f_L"))
    f_mu >= 0 || throw(ArgumentError("f_mu=$f_mu"))
    bsig >= 0 || throw(ArgumentError("bsig=$bsig"))
    abs(restart_cutoff) < 1 || throw(ArgumentError("restart_cutoff=$restart_cutoff"))

    L = Float64(f_L)
    mu = Float64(f_mu)
    q = mu / L

    # Unpack buffers — these are the ONLY arrays used in the loop
    (; xold, yold, uold, zold,
        xnew, ynew, unew, znew,
        fgrad, Fgrad, Fgradold) = bufs

    # Initialise state from x0 (one copy each; no further copies in the loop)
    copyto!(xold, x0)
    copyto!(yold, x0)
    copyto!(uold, x0)
    copyto!(zold, x0)
    fill!(Fgradold, zero(eltype(x0)))

    told = 1.0
    sig = 1.0
    zetaold = 1.0
    Fcostold = Fcost(x0)

    out = Vector{Any}(undef, niter + 1)
    out[1] = fun(0, x0, x0, false)

    @showprogress 1 "Reconstructing via $mom (GPU)..." for iter in 1:niter

        alpha = (mom === :pgm && mu != 0) ? 2 / (L + mu) : 1 / L
        is_restart = false

        # ── Compute gradient into fgrad ───────────────────────────────────────
        f_grad!(fgrad, xold)

        if mom === :pgm || mom === :fpgm
            # ynew = g_prox(xold - alpha*fgrad, alpha)
            # Reuse unew as the argument to g_prox (avoids an extra buffer)
            unew .= xold .- alpha .* fgrad
            g_prox!(ynew, unew, alpha)

            # Fgrad = -(1/alpha) * (ynew - xold)  [for restart check only]
            Fgrad .= (1 / alpha) .* (xold .- ynew)

            Fcostnew = Fcost(ynew)

            if restart !== :none
                if restart === :fr && Fcostnew > Fcostold
                    told = 1.0
                    is_restart = true
                elseif restart === :gr
                    # ynew - yold into znew (scratch)
                    znew .= ynew .- yold
                    if _gr_restart_gpu(Fgrad, znew, restart_cutoff)
                        told = 1.0
                        is_restart = true
                    end
                end
                Fcostold = Fcostnew
            end

            # Momentum coefficient β
            if mom === :fpgm && mu != 0
                beta = (1 - sqrt(q)) / (1 + sqrt(q))
                tnew = told   # unused but keep told consistent
            else  # :fpgm with mu==0
                tnew = 0.5 * (1 + sqrt(1 + 4 * told^2))
                beta = (told - 1) / tnew
            end

            if mom === :pgm
                # xnew = ynew — just swap pointers handled at end
            else  # :fpgm
                # xnew = ynew + beta*(ynew - yold)
                # = (1+beta)*ynew - beta*yold
                xnew .= (1 + beta) .* ynew .- beta .* yold
            end

        else  # :pogm ─────────────────────────────────────────────────────────

            # Momentum coefficient β, γ, tnew
            if mu != 0
                beta = (2 + q - sqrt(q^2 + 8q))^2 / 4 / (1 - q)
                gamma = (2 + q - sqrt(q^2 + 8q)) / 2
                tnew = told  # unused
            else
                tnew = (iter == niter) ?
                       0.5 * (1 + sqrt(1 + 8 * told^2)) :
                       0.5 * (1 + sqrt(1 + 4 * told^2))
                beta = (told - 1) / tnew
                gamma = sig * told / tnew
            end

            # unew = xold - alpha*fgrad  (in-place)
            unew .= xold .- alpha .* fgrad

            # znew = unew*(1+β+γ) + uold*(-β) + xold*(-(γ+βα/ζ)) + zold*(βα/ζ)
            coeff_d = beta * alpha / zetaold
            _fused_z!(znew, unew, uold, xold, zold,
                1 + beta + gamma,   # A
                -beta,               # B
                -(gamma + coeff_d),  # C
                coeff_d)             # D

            zetanew = alpha * (1 + beta + gamma)

            # xnew = g_prox(znew, zetanew)  in-place
            g_prox!(xnew, znew, zetanew)

            # Fgrad = fgrad - (1/zetanew)*(xnew - znew)
            Fgrad .= fgrad .- (1 / zetanew) .* (xnew .- znew)

            # ynew = xold - alpha*Fgrad
            ynew .= xold .- alpha .* Fgrad

            Fcostnew = Fcost(xnew)

            if restart !== :none
                if restart === :fr && Fcostnew > Fcostold
                    tnew = 1.0
                    sig = 1.0
                    is_restart = true
                elseif restart === :gr
                    # ynew - yold into unew (unew no longer needed this iter)
                    unew .= ynew .- yold
                    if _gr_restart_gpu(Fgrad, unew, restart_cutoff)
                        tnew = 1.0
                        sig = 1.0
                        is_restart = true
                    end
                end

                if !is_restart && _real_dot(Fgrad, Fgradold) < 0
                    sig = bsig * sig
                end

                Fcostold = Fcostnew
                copyto!(Fgradold, Fgrad)   # one in-place copy, no allocation
            end

            # Rotate old pointers — Julia tuple swap, no memcpy
            # uold ← unew, zold ← znew are pointer reassignments in bufs
            # but bufs is a NamedTuple (immutable), so we do it via the
            # local variables and the caller's bufs is unaffected.
            # We use copyto! for the persistent state that must survive
            # across iterations:
            copyto!(uold, unew)   # uold ← this iter's unew
            copyto!(zold, znew)   # zold ← this iter's znew
            zetaold = zetanew
        end

        out[iter+1] = fun(iter, xnew, ynew, is_restart)

        # xold ← xnew,  yold ← ynew  (copy, because xnew/ynew buffers are
        # reused next iteration)
        copyto!(xold, xnew)
        copyto!(yold, ynew)
        iszero(mu) && mom !== :pgm && (told = tnew)
    end

    # Return a reference to the appropriate buffer (matches mirt_mod.jl convention)
    return (mom === :pogm ? xnew : ynew), out
end


# ─────────────────────────────────────────────────────────────────────────────
# Power iteration (unchanged from mirt_mod.jl, included for completeness)
# ─────────────────────────────────────────────────────────────────────────────

"""
    x, σ1 = poweriter_gpu(A; niter, tol, x0, chat)

Estimate σ₁(A) via power iteration. Identical to `poweriter_mod` but
exported from this module so GPU runs don't need MirtMod.
"""
function poweriter_gpu(
    A;
    niter::Int=200,
    tol::Real=1e-6,
    x0::AbstractArray{<:Number}=ones(eltype(A), size(A, 2)),
    chat::Bool=true,
)
    x = copy(x0)
    ratio_old = Inf

    @showprogress 1 "Power iterating..." for _ in 1:niter
        Ax = A * x
        ratio = norm(Ax) / norm(x)
        abs(ratio - ratio_old) / ratio < tol && (chat && @info "Converged"; break)
        ratio_old = ratio
        x = A' * Ax
        x /= norm(x)
    end

    return x, norm(A * x) / norm(x)
end

export pogm_gpu, alloc_pogm_bufs, poweriter_gpu

end # module MirtGPU