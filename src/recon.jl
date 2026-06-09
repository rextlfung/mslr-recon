module Recon

export img2patches, patches2img, patch_nucnorm, SVST, patchSVST

#=
recon.jl
Core library for iterative fMRI reconstruction via locally low-rank (LLR) regularization.

Contents (operates on a 4-D image time series (Nx,Ny,Nz,Nt) → (space × time) patches):
  - Patch extraction and recombination: img2patches / patches2img
  - Nuclear-norm cost functions: patch_nucnorm
  - Singular Value Soft-Thresholding (SVST) proximal operators: SVST / patchSVST

GPU compatibility notes:
  - img2patches / patches2img use `similar` for allocation so they work with
    both CPU Arrays and GPU CuArrays.
  - patchSVST has separate Array (CPU, @threads, full patch tensor) and
    AbstractArray (GPU, streaming per-patch) dispatches. The streaming path
    avoids allocating the full (space × time × Np) tensor, which can exceed
    10 GB for moderate patch sizes (e.g. 6³ patches over a 90×90×60 volume).
  - SVST dispatches on array type for the reconstruction step: CPU uses findall
    to skip zero singular values; GPU uses a full matrix multiply (findall on
    GPU triggers slow scalar indexing).

Rex Fung, University of Michigan
=#

using Base.Threads
using LinearAlgebra


# ============================================================
# 3-D Patch Operators
# ============================================================

"""
    img2patches(img, patch_size, stride_size) -> P

Extract (space × time) patches from a 4-D image time series `(Nx, Ny, Nz, Nt)`.
Works on both CPU Arrays and GPU CuArrays.

# Returns
- `P`: 3-D array of size `(prod(patch_size), Nt, Np)`, same array type as `img`
"""
function img2patches(img::AbstractArray{<:Any,4}, patch_size, stride_size)
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz   = patch_size
    ssx, ssy, ssz   = stride_size

    all(>(0), stride_size) || throw(ArgumentError("stride_size elements must be positive, got $stride_size"))
    psx = min(psx, Nx); psy = min(psy, Ny); psz = min(psz, Nz)

    Nsteps_x = cld(Nx - psx, ssx)
    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)
    Np = (Nsteps_x + 1) * (Nsteps_y + 1) * (Nsteps_z + 1)

    # Use similar so CuArrays produce CuArrays, Arrays produce Arrays
    P = fill!(similar(img, ComplexF32, psx * psy * psz, Nt, Np), zero(ComplexF32))

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y, ix in 0:Nsteps_x
        sx = min(ix*ssx + 1, Nx - psx + 1)
        sy = min(iy*ssy + 1, Ny - psy + 1)
        sz = min(iz*ssz + 1, Nz - psz + 1)
        patch = view(img, sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1, :)
        P[:, :, ip] .= reshape(patch, psx * psy * psz, Nt)
        ip += 1
    end
    return P
end


"""
    patches2img(P, patch_size, stride_size, og_size) -> img

Recombine (space × time) patches into a 4-D image via overlap-averaging.
Works on both CPU Arrays and GPU CuArrays.

# Arguments
- `P`:        3-D array `(prod(patch_size), Nt, Np)`
- `og_size`:  3-tuple `(Nx, Ny, Nz)` — original spatial dimensions

# Returns
- `img`: 4-D array `(Nx, Ny, Nz, Nt)`, same array type as `P`
"""
function patches2img(P::AbstractArray, patch_size, stride_size, og_size::NTuple{3};
                     pcount=nothing)
    _, Nt, _ = size(P)
    psx, psy, psz = patch_size
    ssx, ssy, ssz = stride_size
    Nx, Ny, Nz    = og_size

    psx = min(psx, Nx); psy = min(psy, Ny); psz = min(psz, Nz)

    Nsteps_x = cld(Nx - psx, ssx)
    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)

    img    = fill!(similar(P, ComplexF32, Nx, Ny, Nz, Nt), zero(ComplexF32))
    Pcount = pcount !== nothing ?
        fill!(pcount, zero(Float32)) :
        fill!(similar(P, Float32, Nx, Ny, Nz), zero(Float32))

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y, ix in 0:Nsteps_x
        sx = min(ix*ssx + 1, Nx - psx + 1)
        sy = min(iy*ssy + 1, Ny - psy + 1)
        sz = min(iz*ssz + 1, Nz - psz + 1)
        patch = reshape(view(P, :, :, ip), psx, psy, psz, Nt)
        img[sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1, :] .+= patch
        Pcount[sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1] .+= 1f0
        ip += 1
    end

    # Replace zeros with 1 to avoid divide-by-zero (max is GPU-friendly)
    Pcount .= max.(Pcount, 1f0)
    img ./= Pcount    # broadcasts (Nx,Ny,Nz) over (Nx,Ny,Nz,Nt)
    return img
end


# ============================================================
# Nuclear-Norm Cost Functions
# ============================================================

"""
    patch_nucnorm(P) -> cost

Sum of nuclear norms across all patches. Each patch matrix is `(space × time)`.
Multi-threaded for CPU Arrays; non-Array inputs (e.g. CuArrays) are moved to CPU first
to avoid multi-threaded CUSOLVER calls, which are not safe and cause OOM errors.
"""
function patch_nucnorm(P::Array)
    @assert ndims(P) == 3 "P must be (space × time × patches)"
    Np = size(P, 3)
    costs = zeros(real(eltype(P)), Np)
    @threads for ip in 1:Np
        costs[ip] = sum(svdvals(view(P, :, :, ip)))
    end
    return sum(costs)
end

function patch_nucnorm(P::AbstractArray)
    return patch_nucnorm(Array(P))
end


# ============================================================
# Singular Value Soft-Thresholding (SVST)
# ============================================================

"""
    SVST(X, β) -> (X_lr, reg)

Singular Value Soft-Thresholding: proximal operator for the nuclear norm.
Shrinks singular values by `β` (zeros those below `β`).
Returns the thresholded matrix and `reg = sum(max.(σ .- β, 0))`, the nuclear
norm of the result, at no extra cost (byproduct of the SVD already computed).

Works for both CPU (`Array`) and GPU (`CuArray`) matrices.
On CPU, zero singular values are skipped for efficiency.
On GPU, a full matrix multiply is used (findall is slow on GPU).
"""
function SVST(X::AbstractMatrix, β)
    # Skip SVD for zero patches to avoid LAPACK SLASCL warnings — CPU only.
    # On GPU, calling norm() launches cuBLAS + a device→host copy per patch,
    # which exhausts CUDA resources when called 486k times (unit-patch scale).
    X isa Array && norm(X) == 0 && return fill!(similar(X), zero(eltype(X))), 0f0

    # DivideAndConquer SVD is fastest on CPU but can fail for ill-conditioned
    # matrices; fall back to QRIteration (CPU only via LAPACK)
    F = try
        svd(X)
    catch e
        X isa Array ? svd(X; alg = LinearAlgebra.QRIteration()) : rethrow(e)
    end

    β_T      = eltype(F.S)(β)
    s_thresh = max.(F.S .- β_T, zero(eltype(F.S)))
    return _svst_reconstruct(F, s_thresh, X), Float32(sum(s_thresh))
end

# CPU: skip zero singular-value columns (faster for sparse spectra)
function _svst_reconstruct(F, s_thresh, ::Array)
    keep = findall(>(0), s_thresh)
    isempty(keep) && return zeros(eltype(F.U), size(F.U, 1), size(F.Vt, 2))
    return F.U[:, keep] * Diagonal(s_thresh[keep]) * F.Vt[keep, :]
end

# GPU / any other AbstractArray: full matrix multiply
# (findall on CuArray triggers slow scalar indexing)
function _svst_reconstruct(F, s_thresh, ::AbstractArray)
    return F.U * Diagonal(s_thresh) * F.Vt
end


# ── CPU / GPU loop dispatch ────────────────────────────────────────────────────

# CPU: multi-threaded over patches; returns sum of per-patch nuclear norms.
function _svst_loop!(P::Array, β, Np)
    costs = zeros(Float32, Np)
    @threads for ip in 1:Np
        P[:, :, ip], costs[ip] = SVST(copy(view(P, :, :, ip)), β)
    end
    return sum(costs)
end

# GPU (or any non-Array AbstractArray): sequential CUSOLVER calls, no @threads
function _svst_loop!(P::AbstractArray, β, Np)
    reg = 0f0
    for ip in 1:Np
        P[:, :, ip], c = SVST(copy(view(P, :, :, ip)), β)
        reg += c
    end
    return reg
end


# Unit patches [1,1,1]: SVST of each (1, Nt) row = block soft-threshold of each voxel's time
# series, vectorized over the whole volume. On GPU this avoids ~Nvox serial cuBLAS / cuSOLVER
# launches that would exhaust CUDA resources within a few POGM iterations. The SVD of a 1×Nt
# row is U=[1], S=[‖x‖], Vᵀ=x/‖x‖, so SVST ⇒ max(1−β/‖x‖,0)·x and nuclear norm = ‖x‖₂.
# IEEE: β/0 = Inf → 1−Inf = −Inf → max(−Inf,0) = 0 for zero voxels.
function _unit_block_svst(img::AbstractArray{<:Any,4}, β)
    norms  = sqrt.(sum(abs2, img; dims=4))          # (Nx,Ny,Nz,1); fused map+reduce, no temp
    β_T    = Float32(β)
    result = img .* max.(1f0 .- β_T ./ norms, 0f0)
    reg    = Float32(sum(max.(dropdims(norms; dims=4) .- β_T, 0f0)))
    return result, reg
end

"""
    patchSVST(img, β, patch_size, stride_size) -> (img_lr, reg)

Apply patch-wise SVST to a 4-D image `(Nx,Ny,Nz,Nt)` with global threshold `β`.
Returns the thresholded image and `reg`, the nuclear norm of the result across all
patches (sum of thresholded singular values), at no extra SVD cost.
CPU (`Array`) path builds the full patch tensor and uses `@threads`.
GPU (`AbstractArray`) path streams one patch at a time — avoids the O(Np) tensor
allocation, which can exceed 10 GB for moderate patch sizes on large volumes.
"""
function patchSVST(img::Array{<:Any,4}, β, patch_size, stride_size; pcount=nothing)
    Nx, Ny, Nz, _ = size(img)
    psx, psy, psz = min(patch_size[1], Nx), min(patch_size[2], Ny), min(patch_size[3], Nz)
    if psx == 1 && psy == 1 && psz == 1
        return _unit_block_svst(img, β)
    end
    P   = img2patches(img, patch_size, stride_size)
    reg = _svst_loop!(P, β, size(P, 3))
    return patches2img(P, patch_size, stride_size, size(img)[1:3]; pcount), reg
end

function patchSVST(img::AbstractArray{<:Any,4}, β, patch_size, stride_size; pcount=nothing)
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz   = patch_size
    ssx, ssy, ssz   = stride_size
    all(>(0), stride_size) || throw(ArgumentError("stride_size elements must be positive, got $stride_size"))
    psx = min(psx, Nx); psy = min(psy, Ny); psz = min(psz, Nz)

    if psx == 1 && psy == 1 && psz == 1
        return _unit_block_svst(img, β)
    end

    Nsteps_x = cld(Nx - psx, ssx)
    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)
    img_out = fill!(similar(img, ComplexF32, Nx, Ny, Nz, Nt), zero(ComplexF32))
    Pcount  = pcount !== nothing ?
        fill!(pcount, zero(Float32)) :
        fill!(similar(img, Float32, Nx, Ny, Nz), zero(Float32))
    reg = 0f0
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y, ix in 0:Nsteps_x
        sx = min(ix*ssx + 1, Nx - psx + 1)
        sy = min(iy*ssy + 1, Ny - psy + 1)
        sz = min(iz*ssz + 1, Nz - psz + 1)
        P_p = reshape(img[sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1, :], psx*psy*psz, Nt)
        result, c = SVST(P_p, β)
        img_out[sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1, :] .+=
            reshape(result, psx, psy, psz, Nt)
        Pcount[sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1] .+= 1f0
        reg += c
    end
    Pcount .= max.(Pcount, 1f0)
    img_out ./= Pcount
    return img_out, reg
end

end # module Recon