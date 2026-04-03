#=
sense_gpu.jl
GPU-native SENSE encoding operator and GPU patch-SVD utilities.

Requires CUDA.jl. Loaded conditionally by reconstruct.jl when use_gpu=true.

Contents:
  - Asense_gpu   : GPU-native drop-in for MIRT.Asense, built on cuFFT + CUDA.jl
  - patchSVST_gpu: Patch-wise SVST using sequential CUSOLVER SVDs (no @threads)

The Asense_gpu operator uses the same fftshift/scale convention as MIRT.Asense
with fft_forward=true, unitary=true, giving σ₁(A) ≈ 1.0 for normalised smaps.

Rex Fung, University of Michigan
=#

module SenseGPU

export Asense_gpu, patchSVST_gpu

using CUDA
using LinearAlgebra
using LinearMapsAA: LinearMapAA
# fft/ifft on CuArrays dispatch to cuFFT via AbstractFFTs when CUDA.jl is loaded.
# fftshift/ifftshift are pure-Julia AbstractArray operations (no special GPU import needed).
using FFTW: fftshift, ifftshift


# ──────────────────────────────────────────────────────────────────────────────
# GPU SENSE encoding operator
# ──────────────────────────────────────────────────────────────────────────────

"""
    Asense_gpu(samp, smaps; fft_forward=true, unitary=true) -> LinearMapAA

GPU-native SENSE encoding operator for a single time frame.
Drop-in replacement for `MIRT.Asense`; returns a `LinearMapAA` with the same
`idim` and `odim` so it works directly with `LinearMapsAA.block_diag`.

# Arguments
- `samp`:  `(Nx, Ny, Nz)` Bool sampling mask (CPU or GPU; converted internally)
- `smaps`: `(Nx, Ny, Nz, Nc)` sensitivity maps as a `CuArray{ComplexF32}`

# Keyword arguments
- `fft_forward`: if `true` (default), forward direction is image → k-space (MRI convention)
- `unitary`:     if `true` (default), scale by `1/√N` to give σ₁(A) ≈ 1

# Returns
A `LinearMapAA` with `idim = (Nx,Ny,Nz)` and `odim = (K, Nc)`.

# Convention
  Forward:  y = scale · fftshift(fft(ifftshift(smaps .* x, D), D), D)[samp, :]
  Adjoint:  x = scale · sum(conj(smaps) .* fftshift(ifft(ifftshift(y_full, D), D), D), dims=D+1)
where D = 1:3 (spatial dims) and scale = 1/√prod(N) when unitary=true.
"""
function Asense_gpu(samp::AbstractArray{Bool}, smaps::CuArray;
                    fft_forward::Bool = true,
                    unitary::Bool     = true)

    N   = size(smaps)[1:end-1]          # spatial dimensions tuple (Nx, Ny, Nz)
    Nc  = size(smaps, ndims(smaps))     # number of coils
    D   = 1:length(N)                   # spatial FFT dimensions
    K   = sum(samp)                     # number of sampled k-space locations
    scale = Float32(unitary ? 1 / sqrt(prod(N)) : 1)

    # Pre-compute linear index array on GPU for fast gather/scatter operations.
    # Int32 saves GPU memory vs Int64.
    idx = CuArray(Int32.(findall(vec(samp))))

    # ── Forward: image (N...) → sampled k-space (K, Nc) ─────────────────────
    function fwd(x_vec::AbstractVector)
        x  = reshape(CuVector{ComplexF32}(x_vec), N...)   # (Nx, Ny, Nz)
        xc = smaps .* x                                    # (Nx, Ny, Nz, Nc)
        if fft_forward
            kc = scale .* fftshift(fft(ifftshift(xc, D), D), D)
        else
            kc = scale .* fftshift(ifft(ifftshift(xc, D), D), D)
        end
        # Gather sampled locations: (prod(N), Nc)[idx, :] → (K, Nc)
        return vec(reshape(kc, :, Nc)[idx, :])             # (K*Nc,)
    end

    # ── Adjoint: sampled k-space (K, Nc) → image (N...) ─────────────────────
    function adj(y_vec::AbstractVector)
        y = reshape(CuVector{ComplexF32}(y_vec), K, Nc)    # (K, Nc)

        # Scatter y back to full k-space grid
        kc_full = CUDA.zeros(ComplexF32, prod(N), Nc)      # (prod(N), Nc)
        kc_full[idx, :] .= y                               # scatter
        kc_full = reshape(kc_full, N..., Nc)               # (Nx, Ny, Nz, Nc)

        if fft_forward
            xc = scale .* ifftshift(ifft(fftshift(kc_full, D), D), D)
        else
            xc = scale .* ifftshift(fft(fftshift(kc_full, D), D), D)
        end

        # SENSE combination: sum over coils with conjugate smaps
        x = dropdims(sum(conj(smaps) .* xc; dims = length(N) + 1); dims = length(N) + 1)
        return vec(x)                                       # (prod(N),)
    end

    return LinearMapAA(fwd, adj, (K * Nc, prod(N));
                       T    = ComplexF32,
                       idim = N,
                       odim = (K, Nc),
                       prop = (name = "Asense_gpu",))
end


# ──────────────────────────────────────────────────────────────────────────────
# GPU patch SVD utilities
# ──────────────────────────────────────────────────────────────────────────────

"""
    SVST_gpu(X, β) -> X_lr

Singular Value Soft-Thresholding for GPU matrices.
Uses a full matrix multiply (no `findall`) which is efficient on GPU.
Dispatches `svd(X)` to CUSOLVER for `CuMatrix` inputs.
"""
function SVST_gpu(X::AbstractMatrix, β)
    norm(X) == 0 && return fill!(similar(X), zero(eltype(X)))
    F        = svd(X)                                      # CUSOLVER SVD for CuMatrix
    β_T      = eltype(F.S)(β)
    s_thresh = max.(F.S .- β_T, zero(eltype(F.S)))
    return F.U * Diagonal(s_thresh) * F.Vt                 # full multiply, no findall
end


"""
    patchSVST_gpu(img, β, patch_size, stride_size) -> img_lr

Patch-wise SVST on GPU with a global threshold `β`.
Uses sequential CUSOLVER SVDs (no CPU `@threads`).
`img` should be a `CuArray{ComplexF32, 4}`.
"""
function patchSVST_gpu(img::AbstractArray, β, patch_size, stride_size)
    P  = img2patches_gpu(img, patch_size, stride_size)
    Np = size(P, 3)
    for ip in 1:Np
        P[:, :, ip] .= SVST_gpu(copy(view(P, :, :, ip)), β)
    end
    return patches2img_gpu(P, patch_size, stride_size, size(img)[1:3])
end


"""
    patchSVST_gpu(img, λs, patch_size, stride_size) -> img_lr

Patch-wise SVST on GPU with per-patch thresholds `λs` (Vector of length `Np`).
"""
function patchSVST_gpu(img::AbstractArray, λs::Vector, patch_size, stride_size)
    P  = img2patches_gpu(img, patch_size, stride_size)
    Np = size(P, 3)
    for ip in 1:Np
        P[:, :, ip] .= SVST_gpu(copy(view(P, :, :, ip)), λs[ip])
    end
    return patches2img_gpu(P, patch_size, stride_size, size(img)[1:3])
end


# ──────────────────────────────────────────────────────────────────────────────
# Internal GPU patch extract / recombine
# (identical logic to recon.jl but allocation uses CUDA.zeros for clarity)
# ──────────────────────────────────────────────────────────────────────────────

function img2patches_gpu(img::AbstractArray, patch_size, stride_size)
    Nx, Ny, Nz, Nt = size(img)
    psx, psy, psz   = patch_size
    ssx, ssy, ssz   = stride_size

    psx = min(psx, Nx); psy = min(psy, Ny); psz = min(psz, Nz)

    Nsteps_x = cld(Nx - psx, ssx)
    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)
    Np = (Nsteps_x + 1) * (Nsteps_y + 1) * (Nsteps_z + 1)

    P = CUDA.zeros(ComplexF32, psx * psy * psz, Nt, Np)

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

function patches2img_gpu(P::AbstractArray, patch_size, stride_size, og_size)
    _, Nt, _ = size(P)
    psx, psy, psz = patch_size
    ssx, ssy, ssz = stride_size
    Nx, Ny, Nz    = og_size

    psx = min(psx, Nx); psy = min(psy, Ny); psz = min(psz, Nz)

    Nsteps_x = cld(Nx - psx, ssx)
    Nsteps_y = cld(Ny - psy, ssy)
    Nsteps_z = cld(Nz - psz, ssz)

    img    = CUDA.zeros(ComplexF32, Nx, Ny, Nz, Nt)
    Pcount = CUDA.zeros(Float32,   Nx, Ny, Nz)

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

    Pcount .= max.(Pcount, 1f0)
    img ./= Pcount
    return img
end

end # module SenseGPU
