
using Random

# Shoe-horn into the legacy DynamicDiscreteSampler API so that we can leverage existing tests
struct WeightBasedSampler
    w::ResizableWeights
end
WeightBasedSampler() = WeightBasedSampler(ResizableWeights(512))

function Base.push!(wbs::WeightBasedSampler, index, weight)
    (index ∈ eachindex(wbs.w) && (index in wbs)) && throw(ArgumentError("Element $index is already present"))
    index > length(wbs.w) && resize!(wbs.w, max(index, 2length(wbs.w)))
    wbs.w[index] = weight
    wbs
end
function Base.append!(wbs::WeightBasedSampler, inds::AbstractVector, weights::AbstractVector)
    axes(inds) == axes(weights) || throw(DimensionMismatch("inds and weights have different axes"))
    min_ind,max_ind = extrema(inds)
    min_ind < 1 && throw(BoundsError(wbs.w, min_ind))
    max_ind > length(wbs.w) && resize!(wbs.w, max(max_ind, 2length(wbs.w)))
    for (i,w) in zip(inds, weights)
        (i ∈ eachindex(wbs.w) && (i in wbs)) && throw(ArgumentError("Element $i is already present"))
        wbs.w[i] = w
    end
    wbs
end
function Base.delete!(wbs::WeightBasedSampler, index)
    (index ∈ eachindex(wbs.w) && index in wbs) || throw(ArgumentError("Element $index is not present"))
    wbs.w[index] = 0
    wbs
end
Base.in(index::Int, wbs::WeightBasedSampler) = 1 <= index <= wbs.w.m[1] && wbs.w.m[index+10523] != 0

Base.rand(rng::AbstractRNG, wbs::WeightBasedSampler) = rand(rng, wbs.w)
Base.rand(rng::AbstractRNG, wbs::WeightBasedSampler, n::Integer) = [rand(rng, wbs.w) for _ in 1:n]

const DynamicDiscreteSampler = WeightBasedSampler
