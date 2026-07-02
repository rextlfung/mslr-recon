#=
sigma1A_tests.jl
Verify that σ₁(A) ≤ 1 for the SENSE operator built from this dataset.

Builds A for a single time frame (same as run_recon but for one frame only),
then runs power iteration.  Single-frame is sufficient because the full
block-diagonal A has σ₁ = max_t σ₁(A_t), and each frame uses the same smaps.

Usage:
    julia tests/sigma1A_tests.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MAT, HDF5
using LinearAlgebra
using LinearMapsAA: undim
using MIRT: Asense
using Statistics: mean

include(joinpath(@__DIR__, "..", "src", "mirt_mod.jl"))
using .MirtMod: poweriter

const FN_KSP   = "/StorageRAID/rexfung/20260409tap/recon/pd_epi_zf.mat" # CHANGE ME: machine-specific data path
const FN_SMAPS = "/StorageRAID/rexfung/20260409tap/recon/smaps_bart.mat" # CHANGE ME: machine-specific data path
const FRAME    = 1   # which time frame to use (all frames share the same smaps)

println("Loading sensitivity maps …")
smaps_raw = ComplexF32.(matread(FN_SMAPS)["smaps"])
smaps     = smaps_raw ./ (sqrt.(sum(abs2.(smaps_raw); dims=4)) .+ eps(Float32))
Nx, Ny, Nz, Nvc = size(smaps)
println("  smaps: $(Nx)×$(Ny)×$(Nz), Nvc=$Nvc")

println("Loading k-space (frame $FRAME only) …")
ksp_raw   = h5read(FN_KSP, "ksp_epi_zf")   # named tuples {real,imag} on HDF5 compound type
ksp_full  = ComplexF32.([complex(k.real, k.imag) for k in ksp_raw])  # (Nx, Ny, Nz, Nvc, Nt)
ksp_frame = ksp_full[:, :, :, :, FRAME]    # (Nx, Ny, Nz, Nvc)

println("Inferring sampling mask …")
Ω_frame = dropdims(any(!iszero, ksp_frame; dims=4); dims=4)  # (Nx, Ny, Nz) Bool
R = (Nx * Ny * Nz) / sum(Ω_frame)
println("  Acquired voxels: $(sum(Ω_frame)) / $(Nx*Ny*Nz)  (R ≈ $(round(R; digits=2)))")

println("Building SENSE operator for frame $FRAME …")
A = Asense(Ω_frame, smaps; fft_forward=true, unitary=true)

println("Running power iteration …")
_, σ1A = poweriter(undim(A); niter=30, tol=1e-6, chat=false)
println()
println("  σ₁(A) = $(round(σ1A; digits=6))")
println()
if σ1A <= 1.0 + 1e-3
    println("✓  σ₁(A) ≤ 1.0 as expected — update σ1A = $(round(σ1A; digits=6)) in the experiment file")
else
    println("✗  σ₁(A) > 1.0 — sensitivity maps are not properly normalized")
end
