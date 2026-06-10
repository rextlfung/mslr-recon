#=
report.jl
Post-reconstruction reporting.

Usage:
    julia scripts/report.jl /path/to/recon_Nscales.mat [--no-components]

Produces (in the same directory as the input .mat):
    <basename>_report.png   — convergence + rel_change + mean_mag + tSNR
    <basename>_report.txt   — parameters and convergence/image-quality stats
    <basename>_scale<k>.png — per-scale mean magnitude (if Nscales > 1)

Rex Fung, University of Michigan
=#

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

using LinearAlgebra, Statistics, Printf
using MAT
using Plots
using MIRTjim: jim
using LaTeXStrings

include(joinpath(@__DIR__, "..", "src", "metrics.jl"))
using .Metrics

_fmt_vec(v) = "[" * join(Int.(v), ", ") * "]"

function _format_summary(; fn_recon, Nx, Ny, Nz, Nt, R, σ1A, L_val,
        Nscales, patch_sizes, strides, λs,
        Niters, Niters_actual, n_restarts,
        used_gpu, device, runtime_s, mom_str, conv_tol, cycle_spin,
        dc_final, reg_final, rel_change_final,
        img_min, img_max, img_mean, img_std,
        mean_tsnr, peak_tsnr)
    device_label = if device !== nothing
        device
    elseif used_gpu === nothing
        "?"
    else
        used_gpu ? "GPU" : "CPU"
    end
    io = IOBuffer()
    println(io, "Reconstruction Report")
    println(io, "="^60)
    println(io, "File: $(basename(fn_recon))")
    println(io)
    println(io, "── Parameters ─────────────────────────────────────────")
    @printf(io, "  Image:        %d × %d × %d,  Nt = %d\n", Nx, Ny, Nz, Nt)
    println(io, "  Device(s):    ", device_label)
    @printf(io, "  Acceleration: R ≈ %.2f\n", R)
    println(io, "  Momentum:     ", mom_str)
    println(io, "  Cycle spin:   ", cycle_spin === nothing ? "?" : (cycle_spin ? "yes" : "no"))
    println(io, "  Nscales:      ", Nscales)
    for k in 1:Nscales
        ps = _fmt_vec(patch_sizes[k])
        st = strides === nothing ? "—" : _fmt_vec(strides[k])
        @printf(io, "    Scale %d: patch = %-16s stride = %-16s λ = %.4f\n",
                k, ps, st, λs[k])
    end
    if isfinite(σ1A);   @printf(io, "  σ₁(A):        %.4f\n", σ1A);   end
    if isfinite(L_val); @printf(io, "  L:            %.4f\n", L_val); end
    if runtime_s !== nothing
        mm, ss = divrem(round(Int, runtime_s), 60)
        @printf(io, "  Wall-clock:   %dm %02ds (%.1f s)\n", mm, ss, runtime_s)
    end
    println(io)
    println(io, "── Convergence ────────────────────────────────────────")
    @printf(io, "  Iterations:   %d / %d", Niters_actual, Niters)
    if Niters_actual < Niters
        @printf(io, "  (early stop, conv_tol = %.1e)\n", conv_tol === nothing ? NaN : conv_tol)
    else
        println(io, "  (ran full schedule)")
    end
    println(io, "  Restarts:     ", n_restarts)
    @printf(io, "  Final dc_cost:  %.4g\n", dc_final)
    @printf(io, "  Final reg_cost: %.4g\n", reg_final)
    @printf(io, "  Final total:    %.4g\n", dc_final + reg_final)
    if rel_change_final !== nothing
        @printf(io, "  Final ‖Δx‖/‖x‖: %.2e\n", rel_change_final)
    end
    println(io)
    println(io, "── Image quality ──────────────────────────────────────")
    @printf(io, "  |X_recon|:    min = %.3g, mean = %.3g, max = %.3g, std = %.3g\n",
            img_min, img_mean, img_max, img_std)
    @printf(io, "  tSNR:         mean = %.2f, peak = %.2f\n", mean_tsnr, peak_tsnr)
    return String(take!(io))
end


function run_report(fn_recon; show_components=true)
    isfile(fn_recon) || error("File not found: $fn_recon")

    prefix = joinpath(dirname(fn_recon), splitext(basename(fn_recon))[1])

    # ── Load ──────────────────────────────────────────────────────────────────
    println("Loading: $fn_recon")
    f = matread(fn_recon)

    X_recon      = f["X_recon"]
    X_components = f["X"]
    dc_costs     = vec(f["dc_costs"])
    reg_costs    = vec(f["reg_costs"])
    restarts     = Bool.(vec(f["restarts"]))
    R            = f["R"]
    Nscales      = Int(f["Nscales"])
    Niters       = Int(f["Niters"])
    λs           = f["lambdas"] isa AbstractArray ? vec(f["lambdas"]) : [f["lambdas"]]
    patch_sizes  = f["patch_sizes"]
    strides      = haskey(f, "strides")     ? f["strides"]            : nothing
    σ1A          = haskey(f, "sigma1A")     ? f["sigma1A"]            : NaN
    L_val        = haskey(f, "L")           ? f["L"]                  : NaN
    used_gpu     = haskey(f, "used_gpu")    ? Bool(f["used_gpu"])     : nothing
    device       = haskey(f, "device")      ? String(f["device"])     : nothing
    runtime_s    = haskey(f, "runtime_s")   ? f["runtime_s"]          : nothing
    mom_str      = haskey(f, "mom")         ? String(f["mom"])        : "?"
    conv_tol     = haskey(f, "conv_tol")    ? f["conv_tol"]           : nothing
    cycle_spin   = haskey(f, "cycle_spin") ? Bool(f["cycle_spin"])   : nothing
    rel_changes  = haskey(f, "rel_changes") ? vec(f["rel_changes"])   : nothing

    Nx, Ny, Nz, Nt = size(X_recon)
    Niters_actual  = length(dc_costs) - 1
    n_restarts     = count(restarts)

    # ── Image stats ───────────────────────────────────────────────────────────
    mag = abs.(X_recon)
    img_min, img_max = extrema(mag)
    img_mean = mean(mag)
    img_std  = std(mag)

    tsnr_map   = tSNR(X_recon)
    finite_pos = filter(x -> isfinite(x) && x > 0, vec(tsnr_map))
    if isempty(finite_pos)
        @warn "tSNR map has no positive finite values — reconstruction may be zero or diverged"
        mean_tsnr = peak_tsnr = 0.0
    else
        mean_tsnr = mean(finite_pos)
        peak_tsnr = maximum(finite_pos)
    end

    rel_change_final = if rel_changes === nothing
        nothing
    else
        finite_rc = filter(isfinite, rel_changes)
        isempty(finite_rc) ? nothing : last(finite_rc)
    end

    # ── Text summary ──────────────────────────────────────────────────────────
    summary = _format_summary(;
        fn_recon, Nx, Ny, Nz, Nt, R, σ1A, L_val,
        Nscales, patch_sizes, strides, λs,
        Niters, Niters_actual, n_restarts,
        used_gpu, device, runtime_s, mom_str, conv_tol, cycle_spin,
        dc_final = dc_costs[end], reg_final = reg_costs[end],
        rel_change_final,
        img_min, img_max, img_mean, img_std,
        mean_tsnr, peak_tsnr,
    )
    print(summary)
    write("$(prefix)_report.txt", summary)

    # ── Report figure (2×2): convergence, rel_change, mean_mag, tSNR ─────────
    p_conv = plotOpt(dc_costs, reg_costs, restarts)

    p_rc = if rel_changes === nothing
        plot(; title = "Relative iterate change (n/a)", legend = false,
               framestyle = :none)
    else
        valid = findall(isfinite, rel_changes)
        p = plot(valid .- 1, rel_changes[valid];
                 xlabel = "Iteration",
                 ylabel = L"\|\Delta x\| / \|x\|",
                 title  = "Relative iterate change",
                 yscale = :log10,
                 lw     = 2,
                 legend = false)
        if conv_tol !== nothing && conv_tol > 0
            hline!(p, [conv_tol]; label = "conv_tol",
                   linestyle = :dash, color = :gray, alpha = 0.7)
        end
        p
    end

    mean_mag = dropdims(mean(mag; dims=4); dims=4)
    p_mag = jim(mean_mag[:, end:-1:1, :]; title = "Mean magnitude", color = :grays)
    p_tsnr = jim(tsnr_map[:, end:-1:1, :];
                 title = "tSNR  (mean=$(round(mean_tsnr; digits=1)), peak=$(round(peak_tsnr; digits=1)))",
                 color = :inferno)

    panels = Any[p_conv, p_rc, p_mag, p_tsnr]
    p_report = plot(panels...; layout = (2, 2), size = (1400, 900))
    display(p_report)
    savefig(p_report, "$(prefix)_report.png")

    # ── Per-scale mean magnitude (separate figures) ───────────────────────────
    if show_components && Nscales > 1
        println("Plotting scale components …")
        for k in 1:Nscales
            comp_mean = dropdims(mean(abs.(X_components[:, :, :, :, k]); dims=4); dims=4)
            p = jim(comp_mean[:, end:-1:1, :];
                    title  = "Scale $k mean magnitude  (patch=$(_fmt_vec(patch_sizes[k])))",
                    color  = :grays,
                    size   = (1400, 700))
            savefig(p, "$(prefix)_scale$(k).png")
        end
    end

    println("Report saved to: $(dirname(fn_recon))")
    return prefix
end


if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS) || ARGS[1] in ("-h", "--help")
        println("Usage: julia scripts/report.jl <recon_file.mat> [--no-components]")
        exit(0)
    end
    run_report(ARGS[1]; show_components = !("--no-components" in ARGS))
end
