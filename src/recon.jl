module Recon

export img2patches, patches2img, patch_nucnorm, SVST, patchSVST,
       nn_viewshare, sense_comb

#=
recon.jl
Core library for iterative fMRI reconstruction via locally low-rank (LLR) regularization.

Contents:
  - Patch extraction and recombination: shared names dispatch on array ndims
      img2patches / patches2img / patchSVST — 4D img (Nx,Ny,Nz,Nt) → 3D spatial patches
      img2patches / patches2img / patchSVST — 3D img (Ny,Nz,Nt)     → 2D spatial patches
  - Nuclear-norm cost functions
  - Singular Value Soft-Thresholding (SVST) proximal operators
  - k-space utilities (nearest-neighbor view-sharing, SENSE combination)

GPU compatibility notes (3D/4D path only):
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
using Statistics
using FFTW


# ============================================================
# 3-D Patch Operators
# ============================================================

"""
    img2patches(img, patch_size, stride_size) -> P

Extract (space × time) patches from a 4-D image time series `(Nx, Ny, Nz, Nt)`.
Works on both CPU Arrays and GPU CuArrays.
A 2-D + time variant dispatches on `img::AbstractArray{<:Any,3}` (see below).

# Returns
- `P`: 3-D array of size `(prod(patch_size), Nt, Np)`, same array type as `img`
"""
function img2patches(img::AbstractArray{<:Any,4}, patch_size, stride_size)
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz   = patch_size
    ssx, ssy, ssz   = stride_size

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
A 2-D + time variant dispatches on `og_size::NTuple{2}` (see below).

# Arguments
- `P`:        3-D array `(prod(patch_size), Nt, Np)`
- `og_size`:  3-tuple `(Nx, Ny, Nz)` — original spatial dimensions

# Returns
- `img`: 4-D array `(Nx, Ny, Nz, Nt)`, same array type as `P`
"""
function patches2img(P::AbstractArray, patch_size, stride_size, og_size::NTuple{3})
    _, Nt, _ = size(P)
    psx, psy, psz = patch_size
    ssx, ssy, ssz = stride_size
    Nx, Ny, Nz    = og_size

    psx = min(psx, Nx); psy = min(psy, Ny); psz = min(psz, Nz)

    Nsteps_x = cld(Nx - psx, ssx)
    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)

    img    = fill!(similar(P, ComplexF32, Nx, Ny, Nz, Nt),  zero(ComplexF32))
    Pcount = fill!(similar(P, Float32,   Nx, Ny, Nz),       zero(Float32))

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

# Per-patch threshold variants
function _svst_loop!(P::Array, λs::Vector, Np)
    costs = zeros(Float32, Np)
    @threads for ip in 1:Np
        P[:, :, ip], costs[ip] = SVST(copy(view(P, :, :, ip)), λs[ip])
    end
    return sum(costs)
end

function _svst_loop!(P::AbstractArray, λs::Vector, Np)
    reg = 0f0
    for ip in 1:Np
        P[:, :, ip], c = SVST(copy(view(P, :, :, ip)), λs[ip])
        reg += c
    end
    return reg
end


"""
    patchSVST(img, β, patch_size, stride_size) -> (img_lr, reg)

Apply patch-wise SVST to a 4-D image `(Nx,Ny,Nz,Nt)` with global threshold `β`.
Returns the thresholded image and `reg`, the nuclear norm of the result across all
patches (sum of thresholded singular values), at no extra SVD cost.
CPU (`Array`) path builds the full patch tensor and uses `@threads`.
GPU (`AbstractArray`) path streams one patch at a time — avoids the O(Np) tensor
allocation, which can exceed 10 GB for moderate patch sizes on large volumes.
A 2-D + time variant dispatches on `img::AbstractArray{<:Any,3}` (see below).
"""
function patchSVST(img::Array{<:Any,4}, β, patch_size, stride_size)
    Nx, Ny, Nz, _ = size(img)
    psx, psy, psz = min(patch_size[1], Nx), min(patch_size[2], Ny), min(patch_size[3], Nz)
    if psx == 1 && psy == 1 && psz == 1
        norms  = sqrt.(sum(abs2, img; dims=4))
        result = img .* max.(1f0 .- Float32(β) ./ norms, 0f0)
        reg    = sum(max.(dropdims(norms; dims=4) .- Float32(β), 0f0))
        return result, Float32(reg)
    end
    P   = img2patches(img, patch_size, stride_size)
    reg = _svst_loop!(P, β, size(P, 3))
    return patches2img(P, patch_size, stride_size, size(img)[1:3]), reg
end

function patchSVST(img::AbstractArray{<:Any,4}, β, patch_size, stride_size)
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz   = patch_size
    ssx, ssy, ssz   = stride_size
    psx = min(psx, Nx); psy = min(psy, Ny); psz = min(psz, Nz)

    # Unit patches: SVST of a (1, Nt) matrix = block soft-threshold of each voxel's
    # time series.  Vectorised over the whole volume — avoids ~486k serial cuBLAS /
    # cuSOLVER launches that exhaust CUDA resources within a few POGM iterations.
    if psx == 1 && psy == 1 && psz == 1
        # norms shape: (Nx, Ny, Nz, 1); sum(abs2, ·; dims) fuses map+reduce (no temp)
        norms  = sqrt.(sum(abs2, img; dims=4))
        β_T    = Float32(β)
        # IEEE: β_T/0 = Inf → 1-Inf = -Inf → max(-Inf,0) = 0 for zero voxels
        result = img .* max.(1f0 .- β_T ./ norms, 0f0)
        reg    = Float32(sum(max.(dropdims(norms; dims=4) .- β_T, 0f0)))
        return result, reg
    end

    Nsteps_x = cld(Nx - psx, ssx)
    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)
    img_out = fill!(similar(img, ComplexF32, Nx, Ny, Nz, Nt), zero(ComplexF32))
    Pcount  = fill!(similar(img, Float32,   Nx, Ny, Nz),       zero(Float32))
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

"""
    patchSVST(img, λs, patch_size, stride_size) -> (img_lr, reg)

Apply patch-wise SVST to a 4-D image with per-patch thresholds `λs` (length `Np` vector).
Returns the thresholded image and `reg`, the nuclear norm of the result, at no extra SVD cost.
CPU (`Array`) path builds the full patch tensor and uses `@threads`.
GPU (`AbstractArray`) path streams one patch at a time.
"""
function patchSVST(img::Array{<:Any,4}, λs::Vector, patch_size, stride_size)
    P   = img2patches(img, patch_size, stride_size)
    reg = _svst_loop!(P, λs, size(P, 3))
    return patches2img(P, patch_size, stride_size, size(img)[1:3]), reg
end

function patchSVST(img::AbstractArray{<:Any,4}, λs::Vector, patch_size, stride_size)
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz   = patch_size
    ssx, ssy, ssz   = stride_size
    psx = min(psx, Nx); psy = min(psy, Ny); psz = min(psz, Nz)
    Nsteps_x = cld(Nx - psx, ssx)
    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)
    img_out = fill!(similar(img, ComplexF32, Nx, Ny, Nz, Nt), zero(ComplexF32))
    Pcount  = fill!(similar(img, Float32,   Nx, Ny, Nz),       zero(Float32))
    reg = 0f0
    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y, ix in 0:Nsteps_x
        sx = min(ix*ssx + 1, Nx - psx + 1)
        sy = min(iy*ssy + 1, Ny - psy + 1)
        sz = min(iz*ssz + 1, Nz - psz + 1)
        P_p = reshape(img[sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1, :], psx*psy*psz, Nt)
        result, c = SVST(P_p, λs[ip])
        img_out[sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1, :] .+=
            reshape(result, psx, psy, psz, Nt)
        Pcount[sx:sx+psx-1, sy:sy+psy-1, sz:sz+psz-1] .+= 1f0
        reg += c
        ip += 1
    end
    Pcount .= max.(Pcount, 1f0)
    img_out ./= Pcount
    return img_out, reg
end


# ============================================================
# k-Space Utilities
# ============================================================

"""
    nn_viewshare(ksp) -> ksp_nn

Nearest-neighbour k-space view sharing along the time dimension.
For each spatial location, unsampled time frames are filled by the
nearest acquired frame. Locations never sampled remain zero.

# Arguments
- `ksp`: 5-D array of size `(Nx, Ny, Nz, Nc, Nt)`
"""
function nn_viewshare(ksp::AbstractArray)
    Nx, Ny, Nz, _, Nt = size(ksp)
    ksp_nn   = zero.(ksp)
    new_grid = 1:Nt

    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        k_vec    = ksp[i, j, k, :, :]
        old_grid = findall(!iszero, k_vec[1, :])
        isempty(old_grid) && continue
        idxs = [argmin(abs.(old_grid .- loc)) for loc in new_grid]
        ksp_nn[i, j, k, :, :] = k_vec[:, idxs]
    end
    return ksp_nn
end


"""
    sense_comb(ksp, smaps) -> img

SENSE-weighted inverse-FT combination to produce an image time series.
Used for zero-filled or view-shared initialisation (not part of the encoding operator).

# Arguments
- `ksp`:   5-D k-space of size `(Nx, Ny, Nz, Nc, Nt)`
- `smaps`: 4-D sensitivity maps of size `(Nx, Ny, Nz, Nc)`
"""
function sense_comb(ksp::AbstractArray, smaps::AbstractArray)
    Nx, Ny, Nz, _, Nt = size(ksp)
    img   = zeros(eltype(ksp), Nx, Ny, Nz, Nt)
    denom = dropdims(sum(abs2.(smaps); dims=4); dims=4) .+ eps(Float32)
    for t in 1:Nt
        frame_mc = fftshift(ifft(ifftshift(ksp[:, :, :, :, t], (1,2,3)), (1,2,3)), (1,2,3))
        img[:, :, :, t] = dropdims(sum(conj.(smaps) .* frame_mc; dims=4); dims=4) ./ denom
    end
    return img
end


# ============================================================
# 2-D Patch Operators — dispatch on 3-D input (Ny, Nz, Nt)
# ============================================================

"""
    img2patches(img, patch_size, stride_size) -> P

Extract 2-D spatial patches from a 3-D image time series `(Ny, Nz, Nt)`.
Dispatches on `ndims(img) == 3`; see also the 4-D variant above.
"""
function img2patches(img::AbstractArray{<:Any,3}, patch_size, stride_size)
    Ny, Nz, Nt = size(img)
    psy, psz   = patch_size
    ssy, ssz   = stride_size

    psy = min(psy, Ny); psz = min(psz, Nz)

    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)
    Np       = (Nsteps_y + 1) * (Nsteps_z + 1)

    P = fill!(similar(img, ComplexF32, psy * psz, Nt, Np), zero(ComplexF32))

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y
        sy = min(iy*ssy + 1, Ny - psy + 1)
        sz = min(iz*ssz + 1, Nz - psz + 1)
        patch = view(img, sy:sy+psy-1, sz:sz+psz-1, :)
        P[:, :, ip] .= reshape(patch, psy * psz, Nt)
        ip += 1
    end
    return P
end


"""
    patches2img(P, patch_size, stride_size, og_size) -> img

Recombine 2-D spatial patches into a 3-D image `(Ny, Nz, Nt)` via overlap-averaging.
Dispatches on `og_size::NTuple{2}`; see also the 4-D variant above.
"""
function patches2img(P::AbstractArray, patch_size, stride_size, og_size::NTuple{2})
    _, Nt, _  = size(P)
    psy, psz  = patch_size
    ssy, ssz  = stride_size
    Ny, Nz    = og_size

    psy = min(psy, Ny); psz = min(psz, Nz)

    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)

    img    = fill!(similar(P, ComplexF32, Ny, Nz, Nt), zero(ComplexF32))
    Pcount = fill!(similar(P, Float32,   Ny, Nz),      zero(Float32))

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y
        sy = min(iy*ssy + 1, Ny - psy + 1)
        sz = min(iz*ssz + 1, Nz - psz + 1)
        patch = reshape(view(P, :, :, ip), psy, psz, Nt)
        img[sy:sy+psy-1, sz:sz+psz-1, :] .+= patch
        Pcount[sy:sy+psy-1, sz:sz+psz-1] .+= 1f0
        ip += 1
    end

    Pcount .= max.(Pcount, 1f0)
    img ./= Pcount
    return img
end


"""
    patchSVST(img, β, patch_size, stride_size) -> (img_lr, reg)

Patch-wise SVST for 2-D + time data `(Ny, Nz, Nt)`.
Patches are normalized by their leading singular value before thresholding
and rescaled afterwards to preserve contrast. Zero patches are skipped.
`reg` is returned as 0 (σ1-rescaling makes the post-threshold nuclear norm
non-trivial to track without an extra SVD).
Dispatches on `ndims(img) == 3`; see also the 4-D GPU-compatible variant above.
"""
function patchSVST(img::AbstractArray{<:Any,3}, β, patch_size, stride_size)
    P  = img2patches(img, patch_size, stride_size)
    Np = size(P, 3)
    σ1s = [opnorm(view(P, :, :, ip)) for ip in 1:Np]
    @threads for ip in 1:Np
        σ1s[ip] == 0 && continue
        P[:, :, ip] ./= σ1s[ip]
        P[:, :, ip], _ = SVST(copy(view(P, :, :, ip)), β)
        σ1_new = opnorm(view(P, :, :, ip))
        σ1_new > 0 && (P[:, :, ip] .*= σ1s[ip] / σ1_new)
    end
    return patches2img(P, patch_size, stride_size, size(img)[1:2]), 0f0
end

end # module Recon