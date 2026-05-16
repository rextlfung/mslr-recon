#=
analyze.jl
Post-reconstruction analysis and visualization.

Usage:
    julia scripts/analyze.jl /path/to/recon_Nscales.mat [--no-components]

Rex Fung, University of Michigan
=#

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

using LinearAlgebra, Statistics
using MAT
using Plots
using MIRTjim: jim
using LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "analysis.jl"))
using .Analysis

const _JIM_SIZE  = (1400, 700)
const _PLOT_SIZE = (1400, 600)
const _PLOTS_DIR = joinpath(@__DIR__, "..", "plots")


function run_analysis(fn_recon; show_components=true)
    isfile(fn_recon) || error("File not found: $fn_recon")

    mkpath(_PLOTS_DIR)
    prefix = joinpath(_PLOTS_DIR, splitext(basename(fn_recon))[1])

    # ── Load ──────────────────────────────────────────────────────────────────
    println("Loading: $fn_recon")
    f = matread(fn_recon)

    X_recon      = f["X_recon"]        # (Nx, Ny, Nz, Nt) complex
    X_components = f["X"]              # (Nx, Ny, Nz, Nt, Nscales) complex
    dc_costs     = f["dc_costs"]
    reg_costs    = f["reg_costs"]
    restarts     = Bool.(f["restarts"])
    R            = f["R"]
    Nscales      = Int(f["Nscales"])
    Niters       = Int(f["Niters"])
    λs           = f["lambdas"]
    patch_sizes  = f["patch_sizes"]

    Nx, Ny, Nz, Nt = size(X_recon)
    println("  Image size: $(Nx)×$(Ny)×$(Nz), Nt=$Nt")
    println("  Nscales=$Nscales,  Niters=$Niters,  R ≈ $(round(R; digits=2))")
    println("  λs = $(round.(λs; digits=3))")

    mag = abs.(X_recon)   # (Nx, Ny, Nz, Nt) Float64


    # ── 1. Convergence ────────────────────────────────────────────────────────
    println("Plotting convergence …")
    p = plotOpt(dc_costs, reg_costs, restarts; plot_size=_PLOT_SIZE)
    display(p)
    savefig(p, "$(prefix)_convergence.png")


    # ── 2. Mean magnitude (anatomy) ───────────────────────────────────────────
    println("Plotting mean magnitude …")
    mean_mag = dropdims(mean(mag; dims=4); dims=4)
    p = jim(mean_mag;
        title  = "Mean magnitude",
        xlabel = L"x", ylabel = L"y",
        color  = :grays,
        size   = _JIM_SIZE)
    display(p)
    savefig(p, "$(prefix)_mean_magnitude.png")


    # ── 3. tSNR map ───────────────────────────────────────────────────────────
    println("Computing tSNR …")
    tsnr_map   = tSNR(X_recon)
    finite_pos = filter(x -> isfinite(x) && x > 0, vec(tsnr_map))
    if isempty(finite_pos)
        @warn "tSNR map has no positive finite values — reconstruction may be zero or diverged"
        mean_tsnr = 0.0
        peak_tsnr = 0.0
    else
        mean_tsnr = mean(finite_pos)
        peak_tsnr = maximum(finite_pos)
    end
    println("  Mean tSNR = $(round(mean_tsnr; digits=2)),  Peak tSNR = $(round(peak_tsnr; digits=2))")

    p = jim(tsnr_map;
        title  = "tSNR  (mean=$(round(mean_tsnr; digits=2)), peak=$(round(peak_tsnr; digits=2)))",
        xlabel = L"x", ylabel = L"y",
        color  = :inferno,
        size   = _JIM_SIZE)
    display(p)
    savefig(p, "$(prefix)_tsnr.png")


    # ── 4. Temporal std map (functional contrast) ─────────────────────────────
    println("Plotting temporal std …")
    tstd_map = dropdims(std(mag; dims=4); dims=4)
    p = jim(tstd_map;
        title  = "Temporal std  (signal variation)",
        xlabel = L"x", ylabel = L"y",
        color  = :viridis,
        size   = _JIM_SIZE)
    display(p)
    savefig(p, "$(prefix)_temporal_std.png")


    # ── 5. Peak-tSNR voxel time course ────────────────────────────────────────
    println("Plotting peak-tSNR voxel time course …")
    peak_idx   = argmax(replace(tsnr_map, NaN => zero(eltype(tsnr_map))))
    ix, iy, iz = Tuple(peak_idx)
    tc         = mag[ix, iy, iz, :]
    p = plot(tc;
        xlabel = "Frame",
        ylabel = "Magnitude",
        title  = "Time course at peak-tSNR voxel ($ix, $iy, $iz)",
        label  = false,
        lw     = 1.5,
        size   = (1400, 450))
    display(p)
    savefig(p, "$(prefix)_timecourse.png")


    # ── 6. Scale component mean images ────────────────────────────────────────
    if show_components && Nscales > 1
        println("Plotting scale components …")
        for k in 1:Nscales
            comp_mean = dropdims(mean(abs.(X_components[:, :, :, :, k]); dims=4); dims=4)
            p = jim(comp_mean;
                title  = "Scale $k mean magnitude  (patch=$(patch_sizes[k]))",
                xlabel = L"x", ylabel = L"y",
                color  = :grays,
                size   = _JIM_SIZE)
            display(p)
            savefig(p, "$(prefix)_scale$(k).png")
        end
    end

    println("Plots saved to: $(_PLOTS_DIR)")
end


if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS) || ARGS[1] in ("-h", "--help")
        println("Usage: julia scripts/analyze.jl <recon_file.mat> [--no-components]")
        exit(0)
    end
    run_analysis(ARGS[1]; show_components = !("--no-components" in ARGS))
end
