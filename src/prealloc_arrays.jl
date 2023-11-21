module PreallocVectors
using InvertedIndices
mutable struct PreallocVector{T} <: AbstractVector{T}
    v::Vector{T}
    n::Int
end

PreallocVector(T, n) = PreallocVector(Vector{T}(undef, n), 0)
PreallocVector{T}() where {T} = PreallocVector(Vector{T}(), 0)
PreallocVector(n) = PreallocVector(Float64, n)
function PreallocVector(v::AbstractVector{T}) where {T}
    pv = PreallocVector(Float64, length(v))
    pv.v .= v
    return pv
end

Base.eltype(::Type{<:PreallocVector{T}}) where {T} = T
Base.IndexStyle(::Type{<:PreallocVector}) = IndexLinear()

Base.:(==)(x::PreallocVector, y::PreallocVector) = x.v == y.v
Base.length(v::PreallocVector) = v.n
Base.size(v::PreallocVector) = (v.n,)
Base.iterate(v::PreallocVector) = iterate(v.v)
Base.iterate(v::PreallocVector, i) = iterate(v.v, i)

Base.copy(v::PreallocVector) = PreallocVector(copy(v.v), v.n)

function Base.resize!(v::PreallocVector, n)
    resize!(v.v, n)
    v.n = n
    return v
end

function Base.getindex(v::PreallocVector, i)
    return v.v[i]
end

function Base.setindex!(v::PreallocVector, x, i)
    v.n = max(v.n, i)
    v.v[i] = x
    return
end
function Base.deleteat!(v::PreallocVector, i::Vector{Int})
    to_keep = v.v[1:v.n][Not(i)]
    v.n = v.n - length(i)
    v.v[1:v.n] .= to_keep
    return to_keep
end
function Base.keepat!(v::PreallocVector, i::Vector{Int})
    v.n = length(i)
    v.v[1:v.n] .= v.v[i]
    return
end
function Base.push!(v::PreallocVector, x)
    if v.n == length(v.v)
        resize!(v.v, 2 * length(v.v))
    end
    v.n += 1
    v.v[v.n] = x
    return
end

function Base.append!(v::PreallocVector, x)
    if v.n + length(x) > length(v.v)
        resize!(v.v, 2 * (v.n + length(x)))
    end
    v.n = v.n + length(x)
    v.v[v.n-length(x)+1:v.n] .= x
    return
end

function Base.empty!(v::PreallocVector)
    v.n = 0
    return
end
function Base.sizehint!(v::PreallocVector, n)
    if n > length(v.v)
        resize!(v.v, n)
    end
    return v
end
export PreallocVector
end