module Recon

export img2patches, patches2img, patch_nucnorm, SVST, patchSVST,
       nn_viewshare, sense_comb,
       img2patches2D, patches2img2D, patchSVST2D

#=
recon.jl
Core library for iterative fMRI reconstruction via locally low-rank (LLR) regularization.

Contents:
  - Patch extraction and recombination (3D and 2D)
  - Nuclear-norm cost functions
  - Singular Value Soft-Thresholding (SVST) proximal operators
  - k-space utilities (nearest-neighbor view-sharing, SENSE combination)

GPU compatibility notes:
  - img2patches / patches2img use `similar` for allocation so they work with
    both CPU Arrays and GPU CuArrays.
  - patchSVST dispatches to _svst_loop! which is @threads for CPU Arrays and
    sequential for any other AbstractArray (e.g. CuArray).
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

Extract (space × time) patches from a 4-D image time series.
Works on both CPU Arrays and GPU CuArrays.

# Arguments
- `img`:         4-D complex array of size `(Nx, Ny, Nz, Nt)`
- `patch_size`:  3-vector, spatial side lengths of each cubic patch
- `stride_size`: 3-vector, stride between patch origins

# Returns
- `P`: 3-D array of size `(prod(patch_size), Nt, Np)`, same array type as `img`
"""
function img2patches(img::AbstractArray, patch_size, stride_size)
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

# Arguments
- `P`:           3-D array of size `(prod(patch_size), Nt, Np)`
- `patch_size`:  3-vector, spatial side lengths
- `stride_size`: 3-vector, strides used during extraction
- `og_size`:     3-vector, original spatial dimensions `(Nx, Ny, Nz)`

# Returns
- `img`: 4-D array of size `(Nx, Ny, Nz, Nt)`, same array type as `P`
"""
function patches2img(P::AbstractArray, patch_size, stride_size, og_size)
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
"""
function patch_nucnorm(P::AbstractArray)
    @assert ndims(P) == 3 "P must be (space × time × patches)"
    Np = size(P, 3)
    costs = zeros(Np)
    @threads for ip in 1:Np
        costs[ip] = sum(svdvals(view(P, :, :, ip)))
    end
    return sum(costs)
end

"""
    patch_nucnorm(P, λs) -> cost

Weighted sum of nuclear norms, one weight per patch.

# Arguments
- `P`:  3-D patch tensor of size `(space, time, Np)`
- `λs`: `Np`-vector of per-patch regularization weights
"""
function patch_nucnorm(P::AbstractArray, λs::Vector)
    @assert ndims(P) == 3 "P must be (space × time × patches)"
    Np = size(P, 3)
    costs = zeros(Np)
    @threads for ip in 1:Np
        svs = svdvals(copy(P[:, :, ip]))
        costs[ip] = svs[1] > 0 ? λs[ip] * sum(svs) : 0.0
    end
    return sum(costs) / Np
end


# ============================================================
# Singular Value Soft-Thresholding (SVST)
# ============================================================

"""
    SVST(X, β) -> X_lr

Singular Value Soft-Thresholding: proximal operator for the nuclear norm.
Shrinks singular values by `β` (zeros those below `β`).

Works for both CPU (`Array`) and GPU (`CuArray`) matrices.
On CPU, zero singular values are skipped for efficiency.
On GPU, a full matrix multiply is used (findall is slow on GPU).
"""
function SVST(X::AbstractMatrix, β)
    # Skip SVD entirely for zero patches (avoids LAPACK SLASCL warning)
    norm(X) == 0 && return fill!(similar(X), zero(eltype(X)))

    # DivideAndConquer SVD is fastest on CPU but can fail for ill-conditioned
    # matrices; fall back to QRIteration (CPU only via LAPACK)
    F = try
        svd(X)
    catch e
        X isa Array ? svd(X; alg = LinearAlgebra.QRIteration()) : rethrow(e)
    end

    β_T      = eltype(F.S)(β)
    s_thresh = max.(F.S .- β_T, zero(eltype(F.S)))
    return _svst_reconstruct(F, s_thresh, X)
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

# CPU: multi-threaded over patches
function _svst_loop!(P::Array, β, Np)
    @threads for ip in 1:Np
        P[:, :, ip] .= SVST(copy(view(P, :, :, ip)), β)
    end
end

# GPU (or any non-Array AbstractArray): sequential CUSOLVER calls, no @threads
function _svst_loop!(P::AbstractArray, β, Np)
    for ip in 1:Np
        P[:, :, ip] .= SVST(copy(view(P, :, :, ip)), β)
    end
end

# Per-patch threshold variants
function _svst_loop!(P::Array, λs::Vector, Np)
    @threads for ip in 1:Np
        P[:, :, ip] .= SVST(copy(view(P, :, :, ip)), λs[ip])
    end
end

function _svst_loop!(P::AbstractArray, λs::Vector, Np)
    for ip in 1:Np
        P[:, :, ip] .= SVST(copy(view(P, :, :, ip)), λs[ip])
    end
end


"""
    patchSVST(img, β, patch_size, stride_size) -> img_lr

Apply patch-wise SVST to a 4-D image with a **global** threshold `β`.
Equivalent to the average of proximal operators for the nuclear norm.
GPU-compatible: dispatches to sequential CUSOLVER SVDs when `img` is a CuArray.
"""
function patchSVST(img::AbstractArray, β, patch_size, stride_size)
    P  = img2patches(img, patch_size, stride_size)
    Np = size(P, 3)
    _svst_loop!(P, β, Np)
    return patches2img(P, patch_size, stride_size, size(img)[1:3])
end

"""
    patchSVST(img, λs, patch_size, stride_size) -> img_lr

Apply patch-wise SVST with **per-patch** thresholds given by `λs` (length `Np` vector).
GPU-compatible.
"""
function patchSVST(img::AbstractArray, λs::Vector, patch_size, stride_size)
    P  = img2patches(img, patch_size, stride_size)
    Np = size(P, 3)
    _svst_loop!(P, λs, Np)
    return patches2img(P, patch_size, stride_size, size(img)[1:3])
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
    Nx, Ny, Nz, Nc, Nt = size(ksp)
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
    Nx, Ny, Nz, Nc, Nt = size(ksp)
    img    = zeros(eltype(ksp), Nx, Ny, Nz, Nt)
    img_mc = fftshift(ifft(ifftshift(ksp), (1, 2, 3)))

    for t in 1:Nt
        num   = sum(conj.(smaps) .* img_mc[:, :, :, :, t]; dims=4)
        denom = sum(abs2.(smaps); dims=4) .+ eps()
        img[:, :, :, t] = dropdims(num ./ denom; dims=4)
    end
    return img
end


# ============================================================
# 2-D Patch Operators (for 2D+time data)
# ============================================================

"""
    img2patches2D(img, patch_size, stride_size) -> P

Extract 2-D spatial patches from a `(Nz, Ny, Nt)` image time series.
"""
function img2patches2D(img::AbstractArray, patch_size, stride_size)
    Nz, Ny, Nt = size(img)
    psz, psy   = patch_size
    ssz, ssy   = stride_size

    Nsteps_z = fld(Nz - psz, ssz)
    Nsteps_y = fld(Ny - psy, ssy)
    Np       = (Nsteps_z + 1) * (Nsteps_y + 1)

    P = fill!(similar(img, ComplexF32, psz * psy, Nt, Np), zero(ComplexF32))

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y
        patch = img[iz*ssz .+ (1:psz), iy*ssy .+ (1:psy), :]
        P[:, :, ip] = reshape(patch, psz * psy, Nt)
        ip += 1
    end
    return P
end


"""
    patches2img2D(P, patch_size, stride_size, og_size) -> img

Recombine 2-D spatial patches into a `(Nz, Ny, Nt)` image via overlap-averaging.
"""
function patches2img2D(P::AbstractArray, patch_size, stride_size, og_size)
    _, Nt, _  = size(P)
    psz, psy  = patch_size
    ssz, ssy  = stride_size
    Nz, Ny    = og_size

    Nsteps_z = fld(Nz - psz, ssz)
    Nsteps_y = fld(Ny - psy, ssy)

    img    = fill!(similar(P, ComplexF32, Nz, Ny, Nt), zero(ComplexF32))
    Pcount = fill!(similar(P, Float32,   Nz, Ny),      zero(Float32))

    ip = 1
    for iz in 0:Nsteps_z, iy in 0:Nsteps_y
        patch = reshape(P[:, :, ip], psz, psy, Nt)
        img[iz*ssz .+ (1:psz), iy*ssy .+ (1:psy), :] .+= patch
        Pcount[iz*ssz .+ (1:psz), iy*ssy .+ (1:psy)] .+= 1f0
        ip += 1
    end

    Pcount .= max.(Pcount, 1f0)
    for t in 1:Nt
        img[:, :, t] ./= Pcount
    end
    return img
end


"""
    patchSVST2D(img, β, patch_size, stride_size) -> img_lr

Patch-wise SVST for 2-D + time data (`(Nz, Ny, Nt)`).
Patches are normalised by their leading singular value before
thresholding and renormalised afterwards to preserve contrast.
"""
function patchSVST2D(img::AbstractArray, β, patch_size, stride_size)
    P  = img2patches2D(img, patch_size, stride_size)
    Np = size(P, 3)

    σ1s = [opnorm(P[:, :, ip]) for ip in 1:Np]
    σ1s[σ1s .== 0] .= eps()
    P ./= reshape(σ1s, 1, 1, :)

    @threads for ip in 1:Np
        P[:, :, ip] = SVST(copy(view(P, :, :, ip)), β)
    end

    σ1s_tmp = [opnorm(P[:, :, ip]) for ip in 1:Np]
    σ1s_tmp[σ1s_tmp .== 0] .= eps()
    P ./= reshape(σ1s_tmp, 1, 1, :)
    P .*= reshape(σ1s, 1, 1, :)

    return patches2img2D(P, patch_size, stride_size, size(img)[1:2])
end

end # module Recon