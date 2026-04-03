module MirtMod

export pogm_mod, poweriter_mod

#=
mirt_mod.jl
Modified MIRT optimisation routines.

Contents:
  1. pogm_mod    – Proximal Optimised Gradient Method (POGM) with restart
  2. poweriter_mod – Power iteration for estimating the spectral norm of a linear operator

Original authors: Donghwan Kim & Jeff Fessler, University of Michigan, 2017
Modified by:      Rex Fung, University of Michigan, 2025
  – Added ProgressMeter progress bars
=#

using LinearAlgebra: norm
using ProgressMeter


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

function _gr_restart(Fgrad, ynew_yold, restart_cutoff)
    return sum(Float64, real(-Fgrad .* ynew_yold)) <=
           restart_cutoff * norm(Fgrad) * norm(ynew_yold)
end


# ──────────────────────────────────────────────────────────────────────────────
# POGM
# ──────────────────────────────────────────────────────────────────────────────

"""
    x, out = pogm_mod(x0, Fcost, f_grad, f_L; kwargs...)

Proximal Optimised Gradient Method (POGM) with optional restart.
Solves the composite minimisation problem

    min_x  F(x) = f(x) + g(x)

where `f` is smooth convex (with Lipschitz gradient constant `f_L`) and
`g` is convex and proximal-friendly.

# Positional arguments
- `x0`     initial guess
- `Fcost`  `Function`: computes `F(x)` (used only when `restart === :fr`)
- `f_grad` `Function`: computes ∇f(x)
- `f_L`    Lipschitz constant of ∇f

# Keyword arguments
| Name              | Default   | Description                                          |
|:------------------|:----------|:-----------------------------------------------------|
| `f_mu`            | `0.`      | Strong convexity parameter of f                      |
| `mom`             | `:pogm`   | Momentum type: `:pogm`, `:fpgm`, `:pgm`              |
| `restart`         | `:gr`     | Restart rule: `:gr` (gradient), `:fr` (function), `:none` |
| `restart_cutoff`  | `0.`      | cos(angle) threshold for gradient restart            |
| `bsig`            | `1`       | γ decrease factor ∈ [0, 1]                           |
| `niter`           | `10`      | Number of iterations                                 |
| `g_prox`          | identity  | `g_prox(z, c)` = prox_{c·g}(z)                      |
| `fun`             | →`undef`  | `fun(iter, xk, yk, is_restart)` called each iteration |

# Returns
- `x`   final iterate (secondary `xk` for POGM, primary `yk` otherwise)
- `out` vector of `fun(...)` results, length `niter+1`

# References
- Kim & Fessler, "Adaptive restart of the optimized gradient method", 2018
- Beck & Teboulle, SIAM J. Imaging Sci., 2009
- Taylor, Hendrickx & Glineur, SIAM J. Opt., 2017
"""
function pogm_mod(
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
    fun::Function         = (iter, xk, yk, is_restart) -> undef,
)
    mom     ∈ (:pgm, :fpgm, :pogm) || throw(ArgumentError("mom=$mom"))
    restart ∈ (:none, :gr, :fr)    || throw(ArgumentError("restart=$restart"))
    f_L >= 0                        || throw(ArgumentError("f_L=$f_L < 0"))
    f_mu >= 0                       || throw(ArgumentError("f_mu=$f_mu < 0"))
    bsig >= 0                       || throw(ArgumentError("bsig=$bsig < 0"))
    abs(restart_cutoff) < 1         || throw(ArgumentError("restart_cutoff=$restart_cutoff"))

    L  = f_L
    mu = f_mu
    q  = mu / L

    # Initialise momentum state
    told = 1;  sig = 1;  zetaold = 1
    xold = x0; yold = x0; uold = x0; zold = x0
    Fcostold  = Fcost(x0)
    Fgradold  = zeros(size(x0))

    out = Array{Any}(undef, niter + 1)
    out[1] = fun(0, x0, x0, false)

    xnew = similar(x0)
    ynew = similar(x0)

    @showprogress 1 "Reconstructing via $mom..." for iter in 1:niter
        alpha = (mom === :pgm && mu != 0) ? 2 / (L + mu) : 1 / L

        fgrad      = f_grad(xold)
        is_restart = false

        if mom === :pgm || mom === :fpgm
            ynew     = g_prox(xold - alpha * fgrad, alpha)
            Fgrad    = -(1 / alpha) * (ynew - xold)
            Fcostnew = Fcost(ynew)

            if restart !== :none
                if (restart === :fr && Fcostnew > Fcostold) ||
                   (restart === :gr && _gr_restart(Fgrad, ynew - yold, restart_cutoff))
                    told       = 1
                    is_restart = true
                end
                Fcostold = Fcostnew
            end
        else  # :pogm
            unew = xold - alpha * fgrad
        end

        # Momentum coefficient β
        if mom === :fpgm && mu != 0
            beta = (1 - sqrt(q)) / (1 + sqrt(q))
        elseif mom === :pogm && mu != 0
            beta = (2 + q - sqrt(q^2 + 8q))^2 / 4 / (1 - q)
        elseif mom !== :pgm
            tnew = (mom === :pogm && iter == niter) ?
                   0.5 * (1 + sqrt(1 + 8told^2)) :
                   0.5 * (1 + sqrt(1 + 4told^2))
            beta = (told - 1) / tnew
        end

        # Momentum update
        if mom === :pgm
            xnew = ynew
        elseif mom === :fpgm
            xnew = ynew + beta * (ynew - yold)
        else  # :pogm
            gamma = (mu != 0) ? (2 + q - sqrt(q^2 + 8q)) / 2 : sig * told / tnew
            znew  = unew + beta * (unew - uold) +
                    gamma * (unew - xold) -
                    beta * alpha / zetaold * (xold - zold)
            zetanew = alpha * (1 + beta + gamma)
            xnew    = g_prox(znew, zetanew)

            Fgrad    = fgrad - (1 / zetanew) * (xnew - znew)
            ynew     = xold - alpha * Fgrad
            Fcostnew = Fcost(xnew)

            if restart !== :none
                if (restart === :fr && Fcostnew > Fcostold) ||
                   (restart === :gr && _gr_restart(Fgrad, ynew - yold, restart_cutoff))
                    tnew       = 1
                    sig        = 1
                    is_restart = true
                elseif sum(Float64, real(Fgrad .* Fgradold)) < 0
                    sig = bsig * sig
                end
                Fcostold = Fcostnew
                Fgradold = Fgrad
            end

            uold    = unew
            zold    = znew
            zetaold = zetanew
        end

        out[iter + 1] = fun(iter, xnew, ynew, is_restart)
        xold = xnew
        yold = ynew
        (mom !== :pgm && iszero(mu)) && (told = tnew)
    end

    return (mom === :pogm ? xnew : ynew), out
end


# ──────────────────────────────────────────────────────────────────────────────
# Power iteration
# ──────────────────────────────────────────────────────────────────────────────

"""
    x, σ1 = poweriter_mod(A; niter, tol, x0, chat)

Estimate the spectral norm `σ1 = ‖A‖₂` via power iteration.

# Arguments
- `A` linear operator supporting `A * x` and `A' * x`

# Keyword arguments
| Name    | Default       | Description                       |
|:--------|:--------------|:----------------------------------|
| `niter` | `200`         | Maximum iterations                |
| `tol`   | `1e-6`        | Relative convergence tolerance    |
| `x0`    | `ones(n)`     | Starting vector                   |
| `chat`  | `true`        | Print convergence message         |

# Returns
- `x`  approximate dominant right singular vector
- `σ1` approximate spectral norm
"""
function poweriter_mod(
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
