#=
kernel_tests.jl
Correctness and CPU/GPU agreement tests for all key compute kernels.

Layers tested (bottom-up):
  1. img2patches / patches2img round-trip (non-overlapping patches)
  2. img2patches / patches2img: CPU Array vs GPU CuArray
  3. SVST: single matrix with ≥1 zero singular value after threshold (exercises findall branch on CPU)
  4. patchSVST: non-unit patches, CPU Array vs GPU CuArray
  5. patchSVST: unit patches [1,1,1] — GPU takes a separate vectorised broadcast path; non-trivial equivalence
  6. Asense forward and adjoint: CPU (FFTW) vs GPU (cuFFT)
  7. Asense adjoint consistency: dot(Ax,y) ≈ dot(x,A'y) on each backend

Usage:
  julia tests/kernel_tests.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "src", "recon.jl"))
using .Recon

using MIRT: Asense

# ── GPU gate ──────────────────────────────────────────────────────────────────
const HAS_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

if HAS_GPU
    include(joinpath(@__DIR__, "..", "src", "sense_gpu.jl"))
    using .SenseGPU: Asense_gpu
    println("GPU detected: ", CUDA.name(CUDA.device()))
else
    println("⚠  No GPU detected — Asense_gpu and CuArray tests will be skipped.")
end

# ── helpers ───────────────────────────────────────────────────────────────────
relerr(a, b) = norm(vec(Array(ComplexF32.(a))) .- vec(Array(ComplexF32.(b)))) /
               (norm(vec(Array(ComplexF32.(b)))) + eps(Float32))

pass_count = 0
fail_count = 0

function check(label, err, tol)
    global pass_count, fail_count
    ok = err < tol
    ok ? (pass_count += 1) : (fail_count += 1)
    tag = ok ? "PASS" : "FAIL"
    println("  [$tag] $label  relerr=$(round(err; sigdigits=3))  tol=$tol")
    return ok
end

# ── Synthetic data ────────────────────────────────────────────────────────────
Random.seed!(42)
Nx, Ny, Nz, Nt, Nc = 8, 8, 4, 6, 3
img_cpu = randn(ComplexF32, Nx, Ny, Nz, Nt)


# ══════════════════════════════════════════════════════════════════════════════
# 1. img2patches / patches2img round-trip (non-overlapping → exact)
# ══════════════════════════════════════════════════════════════════════════════
println("\n=== 1. img2patches/patches2img round-trip (non-overlapping) ===")
ps1, ss1 = [4, 4, 2], [4, 4, 2]   # tiles exactly: (8/4)×(8/4)×(4/2) = 8 patches
P1     = img2patches(img_cpu, ps1, ss1)
img_rt = patches2img(P1, ps1, ss1, (Nx, Ny, Nz))
check("CPU round-trip (non-overlapping patches)", relerr(img_rt, img_cpu), 1f-5)


# ══════════════════════════════════════════════════════════════════════════════
# 2. img2patches / patches2img: CPU Array vs GPU CuArray
# ══════════════════════════════════════════════════════════════════════════════
println("\n=== 2. img2patches / patches2img CPU vs GPU ===")
if HAS_GPU
    img_gpu   = cu(img_cpu)
    P1_gpu    = img2patches(img_gpu, ps1, ss1)
    img_rt_g  = patches2img(P1_gpu, ps1, ss1, (Nx, Ny, Nz))
    check("img2patches  CPU vs GPU", relerr(P1_gpu, P1),       1f-5)
    check("patches2img  CPU vs GPU", relerr(img_rt_g, img_rt), 1f-5)
else
    println("  [SKIP] GPU not available")
end


# ══════════════════════════════════════════════════════════════════════════════
# 3. SVST: matrix with ≥1 zero singular value post-threshold
#    Exercises the CPU findall(>(0), …) fast path vs GPU full-multiply path.
# ══════════════════════════════════════════════════════════════════════════════
println("\n=== 3. SVST — singular value zeroed after threshold ===")
Random.seed!(7)
M_cpu = randn(ComplexF32, 4, 6)
sv    = svdvals(M_cpu)                         # descending order
β_sv  = Float32(0.5 * (sv[end-1] + sv[end]))  # zeros the smallest singular value
res_svst_cpu, _ = SVST(M_cpu, β_sv)

if HAS_GPU
    res_svst_gpu, _ = SVST(cu(M_cpu), β_sv)
    check("SVST CPU vs GPU", relerr(res_svst_gpu, res_svst_cpu), 1f-4)
else
    println("  [SKIP] GPU not available")
end


# ══════════════════════════════════════════════════════════════════════════════
# 4. patchSVST — non-unit patches
#    CPU: img2patches → @threads _svst_loop! → patches2img
#    GPU: streaming per-patch SVST via AbstractArray dispatch
# ══════════════════════════════════════════════════════════════════════════════
println("\n=== 4. patchSVST — non-unit patches ===")
Random.seed!(13)
β_p = 0.1f0
ps4, ss4 = [4, 4, 2], [4, 4, 2]
res4_cpu, _ = patchSVST(img_cpu, β_p, ps4, ss4)

if HAS_GPU
    res4_gpu, _ = patchSVST(cu(img_cpu), β_p, ps4, ss4)
    check("patchSVST non-unit  CPU vs GPU", relerr(res4_gpu, res4_cpu), 1f-4)
else
    println("  [SKIP] GPU not available")
end


# ══════════════════════════════════════════════════════════════════════════════
# 5. patchSVST — unit patches [1,1,1]
#    CPU: img2patches → SVST of each 1×Nt matrix (SVD path)
#    GPU: vectorised broadcast max.(1 - β/‖x‖, 0)·x  (no SVD!)
#    Equivalence: SVD of 1×Nt row vector gives U=[[1]], S=[‖x‖], Vt=x/‖x‖
#    → SVST result = max(‖x‖-β,0)/‖x‖ · x = max(1-β/‖x‖,0) · x  ✓
# ══════════════════════════════════════════════════════════════════════════════
println("\n=== 5. patchSVST — unit patches [1,1,1] (different code paths!) ===")
ps5, ss5 = [1, 1, 1], [1, 1, 1]
res5_cpu, _ = patchSVST(img_cpu, β_p, ps5, ss5)

if HAS_GPU
    res5_gpu, _ = patchSVST(cu(img_cpu), β_p, ps5, ss5)
    check("patchSVST unit [1,1,1]  CPU vs GPU", relerr(res5_gpu, res5_cpu), 1f-4)
else
    println("  [SKIP] GPU not available")
end


# ══════════════════════════════════════════════════════════════════════════════
# 6 & 7. Asense forward / adjoint: CPU (FFTW) vs GPU (cuFFT)
#        + adjoint self-consistency on each backend: dot(Ax,y) ≈ dot(x,A'y)
# ══════════════════════════════════════════════════════════════════════════════
println("\n=== 6 & 7. Asense forward/adjoint: CPU vs GPU + adjoint consistency ===")
Random.seed!(99)
smaps_cpu = randn(ComplexF32, Nx, Ny, Nz, Nc)
smaps_cpu ./= (sqrt.(sum(abs2.(smaps_cpu); dims=4)) .+ eps(Float32))

samp = zeros(Bool, Nx, Ny, Nz)
samp[1:2:end, :, :] .= true          # ~50% sampling
K = sum(samp)

A_cpu = Asense(samp, smaps_cpu; fft_forward=true, unitary=true)

# MIRT.Asense is a LinearMapAO: inputs must match idim=(Nx,Ny,Nz), outputs match odim=(K,Nc).
x_cpu = randn(ComplexF32, Nx, Ny, Nz)
y_cpu = randn(ComplexF32, K, Nc)

Ax_cpu  = A_cpu * x_cpu        # shape (K, Nc)
Aty_cpu = A_cpu' * y_cpu       # shape (Nx, Ny, Nz)

# CPU adjoint self-consistency: <Ax, y> == <x, A'y>
lhs_c = dot(vec(Ax_cpu), vec(y_cpu))
rhs_c = dot(vec(x_cpu),  vec(Aty_cpu))
adj_err_c = abs(lhs_c - rhs_c) / (abs(lhs_c) + eps(Float32))
check("Asense adjoint consistency (CPU)", adj_err_c, 1f-4)

if HAS_GPU
    smaps_gpu = cu(smaps_cpu)
    A_gpu = Asense_gpu(samp, smaps_gpu; fft_forward=true, unitary=true)

    # Asense_gpu also uses idim/odim, so pass shaped arrays.
    x_gpu   = cu(x_cpu)          # (Nx, Ny, Nz) CuArray
    y_gpu   = cu(y_cpu)          # (K, Nc) CuArray
    Ax_gpu  = A_gpu * x_gpu     # (K, Nc)
    Aty_gpu = A_gpu' * y_gpu    # (Nx, Ny, Nz)

    check("Asense forward  CPU vs GPU", relerr(Ax_gpu,  Ax_cpu),  1f-4)
    check("Asense adjoint  CPU vs GPU", relerr(Aty_gpu, Aty_cpu), 1f-4)

    # GPU adjoint self-consistency
    lhs_g = dot(vec(Array(Ax_gpu)), vec(Array(y_gpu)))
    rhs_g = dot(vec(Array(x_gpu)),  vec(Array(Aty_gpu)))
    adj_err_g = abs(lhs_g - rhs_g) / (abs(lhs_g) + eps(Float32))
    check("Asense adjoint consistency (GPU)", adj_err_g, 1f-4)
end


# ── Summary ───────────────────────────────────────────────────────────────────
total = pass_count + fail_count
println("\n$pass_count/$total tests passed.")
!HAS_GPU && println("(GPU tests were skipped.)")
fail_count > 0 && exit(1)
