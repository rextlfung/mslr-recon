#=
analyze.jl
Post-reconstruction analysis and visualisation.

Usage:
    julia scripts/analyze.jl /path/to/recon_1scales.mat

Optional flags:
    --no-detrend        skip linear drift removal
    --show-components   plot individual scale components (multi-scale recons only)

Rex Fung, University of Michigan
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra, Statistics
using MAT
using Plots
using MIRTjim: jim
using LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "analysis.jl"))
using .Analysis

# ── Parse arguments ───────────────────────────────────────────────────────
if isempty(ARGS) || ARGS[1] in ("-h", "--help")
    println("Usage: julia scripts/analyze.jl <recon_file.mat> [--no-detrend] [--show-components]")
    exit(0)
end

fn_recon          = ARGS[1]
DO_DETREND        = !("--no-detrend" in ARGS)
SHOW_COMPONENTS   = "--show-components" in ARGS

isfile(fn_recon) || error("File not found: $fn_recon")


# ── Load reconstruction ───────────────────────────────────────────────────
println("Loading: $fn_recon")
f = matread(fn_recon)

# X_recon is the summed image (Nx, Ny, Nz, Nt); X is the 5-D components array
X_recon     = f["X_recon"]
X_components = f["X"]          # (Nx, Ny, Nz, Nt, Nscales)
dc_costs    = f["dc_costs"]
reg_costs   = f["reg_costs"]
restarts    = Bool.(f["restarts"])
R           = f["R"]
Nscales     = Int(f["Nscales"])
Niters      = Int(f["Niters"])
λs          = f["lambdas"]
patch_sizes = f["patch_sizes"]

println("  Reconstructed image size: ", size(X_recon))
println("  Nscales = $Nscales,  R ≈ ", round(R; digits=2))
println("  λs = ", round.(λs; digits=3))


# ── Optional detrending ───────────────────────────────────────────────────
if DO_DETREND
    println("Detrending …")
    detrend!(X_recon)
end


# ── Convergence plot ──────────────────────────────────────────────────────
println("Plotting convergence …")
plotOpt(dc_costs, reg_costs, restarts)


# ── tSNR map ──────────────────────────────────────────────────────────────
println("Computing tSNR …")
tsnr_map  = tSNR(X_recon)
nonzero   = filter(>(0), tsnr_map)
mean_tsnr = mean(nonzero)
peak_tsnr = maximum(nonzero)

println("  Mean tSNR = $(round(mean_tsnr; digits=2))")
println("  Peak tSNR = $(round(peak_tsnr; digits=2))")

jim(tsnr_map;
    xlabel = L"x",
    ylabel = L"y",
    color  = :inferno,
    title  = "tSNR  (mean=$(round(mean_tsnr; digits=2)), peak=$(round(peak_tsnr; digits=2)))")


# ── Per-scale component images (optional) ────────────────────────────────
if SHOW_COMPONENTS && Nscales > 1
    for k in 1:Nscales
        jim(X_components[:, :, :, 1, k];
            title = "Scale $k component, frame 1  (patch=$(patch_sizes[k]))")
    end
end
