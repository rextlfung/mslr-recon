module ReconCache

using MAT

export params_match

function _to_mat(v::AbstractVector)
    # MAT.jl round-trips a length-1 Vector{Vector{Int}} (single-scale PATCH_SIZES,
    # e.g. [[6,6,6]]) as Vector{Any}, which fails a Vector{<:Vector} dispatch even
    # though every element is itself a vector — broadened to a runtime check so
    # single-scale configs compare correctly instead of silently erroring (caught
    # by params_match's try/catch and misreported as "different parameters").
    all(x -> x isa AbstractVector, v) ? hcat(v...) : v
end
_to_mat(m::AbstractMatrix) = m

function params_match(fn; NITERS, PATCH_SIZES, STRIDES, σ1A, mom, conv_tol=1e-5, lambda_global=1.0, cycle_spin=false)
    isfile(fn) || return false
    local f
    try; f = matopen(fn); catch; return false; end
    try
        Int(read(f, "Niters"))  == NITERS              || return false
        Int(read(f, "Nscales")) == length(PATCH_SIZES) || return false
        read(f, "sigma1A")      ≈  σ1A                 || return false
        read(f, "mom")          == String(mom)          || return false
        read(f, "conv_tol")     ≈  conv_tol            || return false
        _to_mat(read(f, "patch_sizes")) == _to_mat(PATCH_SIZES) || return false
        _to_mat(read(f, "strides"))     == _to_mat(STRIDES)     || return false
        if haskey(f, "lambda_global")
            read(f, "lambda_global") ≈ lambda_global || return false
        elseif haskey(f, "lambda_scale")
            read(f, "lambda_scale") ≈ lambda_global || return false
        else
            lambda_global ≈ 1.0 || return false
        end
        if haskey(f, "cycle_spin")
            Bool(read(f, "cycle_spin")) == cycle_spin || return false
        else
            cycle_spin == false || return false
        end
        return true
    catch
        return false
    finally
        close(f)
    end
end

end # module ReconCache
