module MirtMod

export pogm_restart, poweriter

#=
mirt_mod.jl
Modified POGM and power iteration: GPU-compatible, convergence-detecting, memory-efficient.

pogm_restart is a modified port of MIRT.pogm_restart (Kim & Fessler, 2017/2018)
with the following changes:
  1. Scalar momentum coefficients (alpha, beta, gamma, zetanew, etc.) are cast to
     real(eltype(x0)) so that Float64 literals do not promote CuArray{ComplexF32}
     to CuArray{ComplexF64}, which would double memory usage and exhaust VRAM.
  2. Fgradold is initialized with similar(x0) instead of zeros(size(x0)), keeping
     the gradient accumulator on the same device as x0.
  3. _gr_restart uses dot() (CUBLAS) instead of real(-Fgrad .* ynew_yold), avoiding
     two large intermediate CuArray allocations per iteration.
  4. znew momentum update uses fused @. broadcast (1 allocation) instead of chained
     arithmetic (3–4 intermediate CuArrays of ~4.2 GB each for a 3-scale
     (90,90,60,387) reconstruction) that would OOM a 48 GB GPU.
  5. Fgrad is pre-allocated and updated in-place; ynew_yold is pre-allocated for
     the _gr_restart call; Fgradold/Fgrad are swapped (not aliased) each iteration.

  6. Early stopping via conv_tol: halts when the relative image-iterate change
     ‖x_new − x_prev‖_F / ‖x_prev‖_F < conv_tol, where x is the proximal-step output
     (ynew for :fpgm/:pgm, xnew for :pogm). Uses already-live buffers (yold/xold), so
     no extra |x|-sized allocation is needed beyond one transient for the norm difference.
     conv_min_iter iterations must elapse before checks begin, and the same grace period
     is re-applied after each gradient restart (since the post-restart step is momentum-free
     and its rel_change is not comparable to the momentum-boosted changes used to set
     conv_tol). The grace period decays by 1 each time a restart fires (floored at 1), so
     a chain of restarts near convergence does not permanently block early stopping.

  7. rel_change is computed every iteration and passed as the 6th argument to `fun`
     (NaN at iter 0). This extends the MIRT.pogm_restart `fun` signature from 5 to 6
     positional args — external callers using the default `fun` are unaffected; custom
     callbacks must accept the new arg.

poweriter wraps MIRT.poweriter, adding a ProgressMeter progress bar.

Original algorithm: Donghwan Kim & Jeff Fessler, University of Michigan, 2017-2018.
Modified by Rex Fung, University of Michigan, 2025.
=#

using LinearAlgebra: norm, dot
using ProgressMeter


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

function _gr_restart(Fgrad, ynew_yold, restart_cutoff)
    # dot(a,b) = sum(conj(a_i)*b_i) via CUBLAS — no intermediate array allocation.
    return -real(dot(vec(Fgrad), vec(ynew_yold))) <=
           restart_cutoff * norm(Fgrad) * norm(ynew_yold)
end


# ──────────────────────────────────────────────────────────────────────────────
# POGM
# ──────────────────────────────────────────────────────────────────────────────

"""
    x, out = pogm_restart(x0, Fcost, f_grad, f_L; kwargs...)

Proximal gradient method with optional momentum (`:pgm`, `:fpgm`, `:pogm`) and restart.
Modified port of `MIRT.pogm_restart`: GPU-compatible, memory-efficient, with early stopping.

Signature follows `MIRT.pogm_restart` except `fun` takes one extra positional arg
(`rel_change` — relative iterate change, NaN at iter 0); see that function's docstring
for everything else.
"""
function pogm_restart(
    x0,
    Fcost::Function,
    f_grad::Function,
    f_L::Real;
    f_mu::Real            = 0.,
    mom::Symbol           = :pogm,
    restart::Symbol       = :gr,
    restart_cutoff::Real  = 0.,
    bsig::Real            = 1,
    niter::Int            = 10,
    g_prox::Function      = (z, c::Real) -> z,
    fun::Function         = (iter, xk, yk, is_restart, Fcostnew, rel_change) -> undef,
    conv_tol::Real        = 0.,
    conv_min_iter::Int    = 10,
)
    mom     ∈ (:pgm, :fpgm, :pogm) || throw(ArgumentError("mom=$mom"))
    restart ∈ (:none, :gr, :fr)    || throw(ArgumentError("restart=$restart"))
    f_L >= 0                        || throw(ArgumentError("f_L=$f_L < 0"))
    f_mu >= 0                       || throw(ArgumentError("f_mu=$f_mu < 0"))
    bsig >= 0                       || throw(ArgumentError("bsig=$bsig < 0"))
    abs(restart_cutoff) < 1         || throw(ArgumentError("restart_cutoff=$restart_cutoff"))

    # Cast all scalars to the array's real element type to prevent Float64 promotion
    # of CuArray{ComplexF32} during momentum arithmetic.
    T = real(eltype(x0))

    L  = T(f_L)
    mu = T(f_mu)
    q  = mu / L

    told = T(1);  sig = T(1);  zetaold = T(1)
    xold = x0; yold = x0; uold = x0; zold = x0
    Fcostold = Fcost(x0)

    out = Array{Any}(undef, niter + 1)
    out[1] = fun(0, x0, x0, false, Fcostold, T(NaN))
    niter == 0 && return x0, out[1:1]
    niter_actual      = niter
    last_restart_iter = 0
    adaptive_cmi      = conv_min_iter   # decays by 1 per restart; see early-stop check below

    # xnew and ynew are assigned inside the loop (not pre-allocated):
    # in :pogm, xnew = g_prox(znew,...) and ynew = @. xold - alpha*Fgrad both rebind
    # immediately, so pre-allocating them would leave 2×4 GB buffers live but unused
    # during g_prox's large patch-tensor allocation, causing OOM on 48 GB GPUs.
    #
    # Fgradold and ynew_yold are only read/written inside the :pogm branch (lines 170-181).
    # For :fpgm/:pgm they alias x0 (already live, zero VRAM overhead) instead of occupying
    # 2×|x0| ≈ 9 GB of VRAM that would prevent the global-scale SVD from allocating its U.
    local xnew, ynew
    Fgrad     = similar(x0)
    Fgradold  = mom === :pogm ? fill!(similar(x0), zero(eltype(x0))) : x0
    ynew_yold = mom === :pogm ? similar(x0) : x0

    @showprogress 1 "Reconstructing via $mom..." for iter in 1:niter
        alpha = (mom === :pgm && mu != 0) ? T(2) / (L + mu) : T(1) / L

        fgrad      = f_grad(xold)
        is_restart = false

        if mom === :pgm || mom === :fpgm
            ynew     = g_prox(xold - alpha * fgrad, alpha)
            Fgrad    = -(T(1) / alpha) * (ynew - xold)
            Fcostnew = Fcost(ynew)

            if restart !== :none
                if (restart === :fr && Fcostnew > Fcostold) ||
                   (restart === :gr && _gr_restart(Fgrad, ynew - yold, restart_cutoff))
                    told       = T(1)
                    is_restart = true
                end
                Fcostold = Fcostnew
            end
        else  # :pogm
            unew = @. xold - alpha * fgrad
        end

        # Momentum coefficient β
        if mom === :fpgm && mu != 0
            beta = (T(1) - sqrt(q)) / (T(1) + sqrt(q))
        elseif mom === :pogm && mu != 0
            beta = (T(2) + q - sqrt(q^2 + T(8)*q))^2 / (T(4) * (T(1) - q))
        elseif mom !== :pgm
            tnew = (mom === :pogm && iter == niter) ?
                   T(0.5) * (T(1) + sqrt(T(1) + T(8)*told^2)) :
                   T(0.5) * (T(1) + sqrt(T(1) + T(4)*told^2))
            beta = (told - T(1)) / tnew
        end

        # Momentum update
        if mom === :pgm
            xnew = ynew
        elseif mom === :fpgm
            xnew = ynew + beta * (ynew - yold)
        else  # :pogm
            gamma   = (mu != 0) ? (T(2) + q - sqrt(q^2 + T(8)*q)) / T(2) : sig * told / tnew
            ba_z    = beta * alpha / zetaold
            # Fused broadcast: 1 allocation instead of 3–4 large intermediates.
            znew    = @. unew + beta * (unew - uold) + gamma * (unew - xold) - ba_z * (xold - zold)
            zetanew = alpha * (T(1) + beta + gamma)
            xnew    = g_prox(znew, zetanew)   # modifies znew in-place; xnew === znew

            iz      = T(1) / zetanew
            @. Fgrad = fgrad - iz * (xnew - znew)
            # fgrad is no longer needed; free it before allocating ynew.
            # Without this, fgrad (4.5 GB) + the 8 other live X-shaped buffers exceed 48 GB VRAM.
            fgrad = nothing
            !(x0 isa Array) && GC.gc(true)
            # Allocating (not in-place): yold = ynew at loop end would corrupt in-place ynew next iter.
            ynew    = @. xold - alpha * Fgrad
            Fcostnew = Fcost(xnew)

            if restart !== :none
                @. ynew_yold = ynew - yold
                if (restart === :fr && Fcostnew > Fcostold) ||
                   (restart === :gr && _gr_restart(Fgrad, ynew_yold, restart_cutoff))
                    tnew       = T(1)
                    sig        = T(1)
                    is_restart = true
                elseif real(dot(vec(Fgrad), vec(Fgradold))) < 0
                    sig = T(bsig) * sig
                end
                Fcostold = Fcostnew
                # Swap buffers: Fgradold, Fgrad point to same device; avoids aliasing corruption.
                Fgradold, Fgrad = Fgrad, Fgradold
            end

            uold    = unew
            zold    = znew
            zetaold = zetanew
        end

        # Relative iterate change — compared between consecutive prox outputs:
        #   FPGM/PGM: ynew vs yold   POGM: xnew vs xold
        # Both reference arrays are already live — no extra |x| buffer needed.
        # norm(curr - prev) allocates one transient; reclaimed by next iter's forced GC.
        # Always computed (logged via fun), and reused for the early-stop check below.
        recon_prev = mom === :pogm ? xold : yold
        recon_curr = mom === :pogm ? xnew : ynew
        rel_change = norm(recon_curr - recon_prev) / (norm(recon_prev) + eps(T))
        if is_restart
            last_restart_iter = iter
            adaptive_cmi      = max(1, adaptive_cmi - 1)
        end

        if conv_tol > 0 && iter >= last_restart_iter + adaptive_cmi && rel_change < T(conv_tol)
            out[iter + 1] = fun(iter, xnew, ynew, is_restart, Fcostnew, rel_change)
            @info "Converged at iteration $iter (‖Δx‖/‖x‖ = $(round(rel_change; sigdigits=2)) < $conv_tol, grace=$adaptive_cmi)"
            niter_actual = iter
            break
        end

        out[iter + 1] = fun(iter, xnew, ynew, is_restart, Fcostnew, rel_change)
        xold = xnew
        yold = ynew
        (mom !== :pgm && iszero(mu)) && (told = tnew)
    end

    return (mom === :pogm ? xnew : ynew), out[1:niter_actual+1]
end


# ──────────────────────────────────────────────────────────────────────────────
# Power iteration
# ──────────────────────────────────────────────────────────────────────────────

"""
    x, σ1 = poweriter(A; niter, tol, x0, chat)

Estimate the spectral norm `σ1 = ‖A‖₂` via power iteration.
Standalone power iteration with a ProgressMeter progress bar.
"""
function poweriter(
    A;
    niter::Int                    = 200,
    tol::Real                     = 1e-6,
    x0::AbstractArray{<:Number}   = ones(eltype(A), size(A, 2)),
    chat::Bool                    = true,
)
    x         = copy(x0)
    ratio_old = Inf

    @showprogress 1 "Power iterating..." for iter in 1:niter
        Ax    = A * x
        ratio = norm(Ax) / norm(x)
        if abs(ratio - ratio_old) / ratio < tol
            chat && @info "Power iteration converged at iter $iter"
            break
        end
        ratio_old = ratio
        x = A' * Ax
        x /= norm(x)
    end

    return x, norm(A * x) / norm(x)
end

end # module MirtMod
