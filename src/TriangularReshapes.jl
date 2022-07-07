module TriangularReshapes

using LoopVectorization

"""
    vector_to_lower_triang!(M::AbstractMatrix{T}, v::AbstractVector{T}) where {T}

Overwrites the lower triangular part of M with the values of v.
# Arguments
- M: The matrix to be overwritten.
- v: The vector to be used to overwrite the lower triangular part of M.
"""
function vector_to_lower_triang!(M::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    n = size(M, 1)
    @assert size(M, 2) == n
    @assert length(v) >= n * (n + 1) / 2
    st = 1
    l = n
    @inbounds @simd for i = 1:n
        @turbo for j = 0:(n - i)
            M[j + i, i] = v[st + j]
        end
        st += l
        l -= 1
    end
    return nothing
end

"""
    vector_to_lower_triang(v::AbstractVector)

Creates a matrix `M`, whose lower triangular part is filled with the values of v.
# Arguments
- v: The vector to be used to overwrite the lower triangular part of M.
# Outputs
- M: The matrix whose lower triangular part is filled with the values of v.
"""
function vector_to_lower_triang(v::AbstractVector{T}) where {T}
    n = length(v)
    nM = Int(-0.5+sqrt(0.25+2*n))
    M = zeros(T, nM, nM)
    vector_to_lower_triang!(M, v)
    return M
end

"""
    lower_triang_to_vector!(v::AbstractVector{T}, M::AbstractMatrix{T}) where {T}

Overwrites the first n * (n + 1) / 2 elements of v with the values of the lower triangular part of M.
# Arguments
- v: The vector to be overwritten.
- M: The matrix whose lower triangular part is used to overwrite the first n * (n + 1) / 2 elements of v.
"""
function lower_triang_to_vector!(v::AbstractVector{T}, M::AbstractMatrix{T}) where {T}
    n = size(M, 1)
    @assert size(M, 2) == n
    @assert length(v) >= n * (n + 1) / 2
    st = 1
    l = n
    @inbounds @simd for i = 1:n
        @turbo for j = 0:(n - i)
            v[st + j] = M[j + i, i]
        end
        st += l
        l -= 1
    end
    return nothing
end

"""
    lower_triang_to_vector(M::AbstractMatrix{T}) where {T}

Returns a vector containing the values of the lower triangular part of M.
# Arguments
- M: The matrix whose lower triangular part is used to create the vector.
"""
function lower_triang_to_vector(M::AbstractMatrix{T}) where {T}
    n = size(M, 1)
    v = Vector{T}(undef, n * (n + 1) ÷ 2)
    lower_triang_to_vector!(v, M)
    return v
end

"""
    lower_triang_to_vector!(v, v1, v2)

Overwrites the first n * (n + 1) / 2 elements of v with the values of the lower triangular part of ``v1 \\cdot v2^\\mathsf{T}``.
# Arguments
- v: The vector to be overwritten min-length: `n * (n + 1) / 2`.
- v1, v2: Vectors of length `n`
"""
function lower_triang_to_vector!(
    v::AbstractVector{T},
    v1::AbstractVector{T},
    v2::AbstractVector{T};
) where {T}
    n = length(v1)
    @assert length(v2) == n
    @assert length(v) >= n * (n + 1) / 2
    l = n
    st = 1
    @inbounds for i = 1:n
        v2i = v2[i]
        @turbo for j = i:n
            v[j - 1 + st] = v1[j] * v2i
        end
        st += l-1
        l -= 1
    end
    return nothing
end

export vector_to_lower_triang!, lower_triang_to_vector!, lower_triang_to_vector, vector_to_lower_triang

end # module TriangularReshapes
