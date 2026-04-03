module Analysis

export tSNR, plotOpt, detrend!

#=
analysis.jl
Post-reconstruction analysis utilities.

Contents:
  - tSNR map computation
  - Optimisation convergence plotting
  - Linear drift detrending

Rex Fung, University of Michigan
=#

using Statistics: mean, std
using LinearAlgebra: pinv
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
# Optimisation convergence plot
# ──────────────────────────────────────────────────────────────────────────────

"""
    plotOpt(dc_costs, reg_costs, restarts; logscale=false)

Plot POGM optimisation progress: data-consistency cost, regularisation cost,
total cost, and restart events.

# Arguments
- `dc_costs`:  data-consistency term per iteration (length `Niters+1`)
- `reg_costs`: regularisation term per iteration
- `restarts`:  boolean vector; `true` indicates a momentum restart at that iteration

# Keyword arguments
- `logscale`: if `true`, use log₁₀ y-axis
"""
function plotOpt(
    dc_costs::Vector,
    reg_costs::Vector,
    restarts::AbstractVector;
    logscale::Bool = false,
)
    Niters = length(dc_costs) - 1
    iters  = 0:Niters

    plt = plot(iters, dc_costs;
        label   = "Data Consistency",
        xlabel  = "Iteration",
        ylabel  = "Cost",
        title   = "POGM Optimisation Convergence",
        lw      = 2,
        legend  = :topright)

    plot!(plt, iters, reg_costs;  label = "Regulariser", lw = 2)
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

    display(plt)
    return plt
end


# ──────────────────────────────────────────────────────────────────────────────
# Detrending
# ──────────────────────────────────────────────────────────────────────────────

"""
    detrend!(img)

Remove voxel-wise linear drift from a 4-D complex image time series in-place.
A least-squares line is fit to the time course of each voxel and the linear
component is subtracted (the mean is preserved).

# Arguments
- `img`: 4-D array of size `(Nx, Ny, Nz, Nt)`
"""
function detrend!(img::AbstractArray{<:Complex})
    Nx, Ny, Nz, Nt = size(img)
    M    = [ones(Nt)  collect(1:Nt)]   # design matrix (intercept + linear ramp)
    Mpinv = pinv(M)

    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        β = Mpinv * vec(img[i, j, k, :])
        img[i, j, k, :] .-= M[:, 2] .* β[2]  # subtract only the linear component
    end
    return img
end

end # module Analysis
