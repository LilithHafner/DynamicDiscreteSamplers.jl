# Shoe-horn into the legacy DynamicDiscreteSampler API so that we can leverage legacy tests and benchmarks
struct WeightBasedSampler
    w::ResizableWeights
end
WeightBasedSampler() = WeightBasedSampler(ResizableWeights(512))

function Base.push!(wbs::WeightBasedSampler, index, weight)
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
        wbs.w[i] = w
    end
    wbs
end
function Base.delete!(wbs::WeightBasedSampler, index)
    index ∈ eachindex(wbs.w) && wbs.w[index] != 0 || throw(ArgumentError("Element $index is not present"))
    wbs.w[index] = 0
    wbs
end
Random.rand!(rng::AbstractRNG, A::AbstractArray, st::Random.SamplerTrivial{<:WeightBasedSampler}) = rand!(rng, A, Random.SamplerTrivial(st[].w))
Random.rand(rng::AbstractRNG, st::Random.SamplerTrivial{<:WeightBasedSampler}) = rand(rng, st[].w)
Random.gentype(::Type{WeightBasedSampler}) = Int

const DynamicDiscreteSampler = WeightBasedSampler
