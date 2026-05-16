#=
sense_gpu.jl
GPU-native SENSE encoding operator.

Requires CUDA.jl. Loaded conditionally by reconstruct.jl when use_gpu=true.

Contents:
  - Asense_gpu: GPU-native drop-in for MIRT.Asense, built on cuFFT + CUDA.jl

Patch operators (patchSVST, img2patches, patches2img, SVST) are NOT duplicated here
because the implementations in src/recon.jl already dispatch correctly for CuArrays:
  - img2patches / patches2img use `similar` → allocates CuArray when input is CuArray.
  - _svst_loop! runs sequentially for AbstractArray (no @threads on GPU).
  - SVST uses a full matrix multiply for AbstractArray (avoids findall on GPU).

The Asense_gpu operator uses the same fftshift/scale convention as MIRT.Asense
with fft_forward=true, unitary=true, giving σ₁(A) ≈ 1.0 for normalized smaps.

Rex Fung, University of Michigan
=#

module SenseGPU

export Asense_gpu

using CUDA
using LinearAlgebra
using LinearMapsAA: LinearMapAA
# fft/ifft are from AbstractFFTs; CUDA.jl registers cuFFT implementations for CuArrays.
# fftshift/ifftshift are pure-Julia AbstractArray operations that work on any array type.
using FFTW: fft, ifft, fftshift, ifftshift


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

    # The adjoint of (scale * fft) is (N * scale * ifft), not (scale * ifft).
    # Julia's fft is the unnormalized DFT; its Hermitian adjoint is N * ifft.
    # So adjoint scale = prod(N) * scale = prod(N)/sqrt(prod(N)) = sqrt(prod(N)).
    adj_scale = Float32(prod(N)) * scale

    # ── Forward: image (N...) → sampled k-space (K, Nc) ─────────────────────
    function fwd(x_vec::AbstractArray)
        x  = reshape(CuVector{ComplexF32}(vec(x_vec)), N...)   # (Nx, Ny, Nz)
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
    function adj(y_vec::AbstractArray)
        y = reshape(CuVector{ComplexF32}(vec(y_vec)), K, Nc)    # (K, Nc)

        # Scatter y back to full k-space grid
        kc_full = CUDA.zeros(ComplexF32, prod(N), Nc)      # (prod(N), Nc)
        kc_full[idx, :] .= y                               # scatter
        kc_full = reshape(kc_full, N..., Nc)               # (Nx, Ny, Nz, Nc)

        if fft_forward
            xc = adj_scale .* ifftshift(ifft(fftshift(kc_full, D), D), D)
        else
            xc = adj_scale .* ifftshift(fft(fftshift(kc_full, D), D), D)
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

end # module SenseGPU
