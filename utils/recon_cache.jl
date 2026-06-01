module ReconCache

using MAT

export params_match

function _to_mat(v::Vector{<:Vector})
    hcat(v...)
end
_to_mat(m::AbstractMatrix) = m

function params_match(fn; NITERS, PATCH_SIZES, STRIDES, σ1A_PRECOMPUTED, mom, conv_tol=1e-5, lambda_scale=1.0)
    isfile(fn) || return false
    local f
    try; f = matopen(fn); catch; return false; end
    try
        Int(read(f, "Niters"))  == NITERS              || return false
        Int(read(f, "Nscales")) == length(PATCH_SIZES) || return false
        read(f, "sigma1A")      ≈  σ1A_PRECOMPUTED    || return false
        read(f, "mom")          == String(mom)          || return false
        read(f, "conv_tol")     ≈  conv_tol            || return false
        _to_mat(read(f, "patch_sizes")) == _to_mat(PATCH_SIZES) || return false
        _to_mat(read(f, "strides"))     == _to_mat(STRIDES)     || return false
        if haskey(f, "lambda_scale")
            read(f, "lambda_scale") ≈ lambda_scale || return false
        else
            lambda_scale ≈ 1.0 || return false
        end
        return true
    catch
        return false
    finally
        close(f)
    end
end

end # module ReconCache
