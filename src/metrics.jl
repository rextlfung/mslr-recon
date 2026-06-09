module Metrics

export tSNR, plotOpt

#=
metrics.jl
Post-reconstruction metrics utilities.

Contents:
  - tSNR map computation
  - Optimization convergence plotting

Rex Fung, University of Michigan
=#

using Statistics: mean, std
using Plots
using LaTeXStrings


# ──────────────────────────────────────────────────────────────────────────────
# tSNR
# ──────────────────────────────────────────────────────────────────────────────

"""
    tSNR(img) -> tSNR_map

Compute the temporal signal-to-noise ratio map of a dynamic image series.
tSNR = mean(|img|) / std(|img|) along the last (time) dimension.

# Arguments
- `img`: N-D complex array; the last axis is treated as time

# Returns
- `tSNR_map`: (N-1)-D real array of the same spatial shape
"""
function tSNR(img::AbstractArray)
    mag = abs.(img)
    N   = ndims(mag)
    ϵ   = eps(eltype(mag))
    return dropdims(mean(mag; dims=N) ./ (std(mag; dims=N) .+ ϵ); dims=N)
end


# ──────────────────────────────────────────────────────────────────────────────
# Optimization convergence plot
# ──────────────────────────────────────────────────────────────────────────────

"""
    plotOpt(dc_costs, reg_costs, restarts; logscale=false)

Plot optimization progress: data-consistency cost, regularization cost,
total cost, and restart events.

# Arguments
- `dc_costs`:  data-consistency term per iteration (length `Niters+1`)
- `reg_costs`: regularization term per iteration
- `restarts`:  boolean vector; `true` indicates a momentum restart at that iteration

# Keyword arguments
- `logscale`: if `true`, use log₁₀ y-axis
"""
function plotOpt(
    dc_costs::Vector,
    reg_costs::Vector,
    restarts::AbstractVector;
    logscale::Bool   = false,
    plot_size::Tuple = (900, 500),
)
    Niters = length(dc_costs) - 1
    iters  = 0:Niters

    plt = plot(iters, dc_costs;
        label   = "Data Consistency",
        xlabel  = "Iteration",
        ylabel  = "Cost",
        title   = "Optimization Convergence",
        lw      = 2,
        legend  = :topright,
        size    = plot_size)

    plot!(plt, iters, reg_costs;  label = "Regularizer", lw = 2)
    plot!(plt, iters, dc_costs .+ reg_costs;
        label     = "Total Cost",
        lw        = 2,
        linestyle = :solid,
        color     = :black)

    restart_iters = findall(restarts) .- 1
    if !isempty(restart_iters)
        vline!(plt, restart_iters;
            label     = "Restart",
            color     = :red,
            linestyle = :dash,
            alpha     = 0.8)
    end

    if logscale
        plot!(plt; yaxis = :log10)
        ylabel!("Cost (log-scale)")
    end

    return plt
end


end # module Metrics
