module DynamicDiscreteSamplers

export DynamicDiscreteSampler, SamplerIndices

#=
julia> @b AliasTable(rand(6)) AliasTables.set_weights!(_, rand!(X))
136.789 ns

julia> @b AliasTable(rand(6)) rand
6.329 ns
=#

using Dictionaries, Distributions, DoubleFloats, Random, StaticArrays

get_weights(p::NTuple{8, Float64}) = p .* typemax(UInt) ./ maximum(p)
get_weights(p::NTuple{8, Any}) = get_weights(Float64.(p))
get_weights(p::NTuple{6, Any}) = get_weights((p..., 0.0, 0.0))
mutable struct RejectionSampler8
    p::NTuple{8, Float64}
    RejectionSampler8(p) = new(get_weights(p))
end

function Random.rand(rng::AbstractRNG, rs::RejectionSampler8)
    while true
        u = rand(rng, UInt64)
        i = u & 0x07 + 1
        u > rs.p[i] || return i
    end
end

function set_weights!(rs::RejectionSampler8, p)
    rs.p = get_weights(p)
    rs
end

# julia> @be RejectionSampler8(rand(NTuple{6, Float64})) rand
# Benchmark: 3322 samples with 1728 evaluations
#  min    10.489 ns
#  median 16.373 ns
#  mean   16.547 ns
#  max    33.275 ns

# julia> @be RejectionSampler8(rand(NTuple{6, Float64})) set_weights!(_, rand(NTuple{6, Float64}))
# Benchmark: 6967 samples with 906 evaluations
#  min    10.946 ns
#  median 12.740 ns
#  mean   14.784 ns
#  max    622.015 ns

all_but_last(x) = x[begin:end-1]
all_but_last(x::SVector) = pop(x)

normalize(x::NTuple) = normalize(SVector(x))
function normalize(p)
    c = cumsum(Float64.(p))
    all_but_last(c) ./ c[end]
end
mutable struct SelectionSampler6
    p::NTuple{5, Float64}
    SelectionSampler6(p) = new(normalize(p))
end
function Base.rand(rng::AbstractRNG, ss::SelectionSampler6)
    u = rand(rng)
    count(<(u), ss.p) + 1
end
function set_weights!(ss::SelectionSampler6, p)
    ss.p = normalize(p)
    ss
end

# julia> @be SelectionSampler6(rand(NTuple{6, Float64})) rand
# Benchmark: 4191 samples with 6525 evaluations
#  min    3.065 ns
#  median 3.608 ns
#  mean   3.472 ns
#  max    6.156 ns

# julia> @be SelectionSampler6(rand(NTuple{6, Float64})) set_weights!(_, rand(NTuple{6, Float64}))
# Benchmark: 4602 samples with 2270 evaluations
#  min    8.150 ns
#  median 8.700 ns
#  mean   9.085 ns
#  max    27.056 ns


mutable struct SelectionSampler{N}
    p::NTuple{N, Float64}
    SelectionSampler(p::NTuple{N, <:Any}) where N = new{N-1}(normalize(p))
end
function Base.rand(rng::AbstractRNG, ss::SelectionSampler)
    u = rand(rng)
    count(<(u), ss.p) + 1
end
function set_weights!(ss::SelectionSampler, p)
    ss.p = normalize(p)
    ss
end

# julia> @b SelectionSampler(rand(NTuple{6, Float64})) rand
# 3.060 ns

# julia> @b SelectionSampler(rand(NTuple{6, Float64})) set_weights!(_, rand(NTuple{6, Float64}))
# 8.368 ns

# julia> @b SelectionSampler(rand(NTuple{64, Float64})) rand
# 16.280 ns

# julia> @b SelectionSampler(rand(NTuple{64, Float64})) set_weights!(_, rand(NTuple{64, Float64}))
# 105.458 μs (3994 allocs: 107.062 KiB)

struct SelectionSampler2{N}
    p::MVector{N, Float64}
    SelectionSampler2(p) = new{length(p)-1}(normalize(p))
end
function Base.rand(rng::AbstractRNG, ss::SelectionSampler2)
    u = rand(rng)
    count(<(u), ss.p) + 1
end
set_weights!(ss::SelectionSampler2, p) = set_weights!(ss, SVector(p))
function set_weights!(ss::SelectionSampler2{T}, p::SVector{U}) where {T, U}
    # cumsum!(ss.p, pop(p))
    # ss.p ./= ss.p[end]+p[end]
    ss.p .= normalize(p)
    ss
end

# julia> @b SelectionSampler2(rand(NTuple{6, Float64})) rand
# 3.033 ns

# julia> @b SelectionSampler2(rand(NTuple{6, Float64})) set_weights!(_, rand(NTuple{6, Float64}))
# 8.186 ns

# julia> @b SelectionSampler2(rand(NTuple{64, Float64})) rand
# 17.953 ns

# julia> @b SelectionSampler2(rand(NTuple{64, Float64})) set_weights!(_, rand(NTuple{64, Float64}))
# 135.489 ns


struct SelectionSampler3{N}
    p::MVector{N, Float64}
end
function Base.rand(rng::AbstractRNG, ss::SelectionSampler3)
    u = rand(rng)*last(ss.p)
    count(<(u), ss.p) + 1
end
set_weights!(ss::SelectionSampler3, p) = set_weights!(ss, SVector(p))
function set_weights!(ss::SelectionSampler3, p::SVector)
    cumsum!(ss.p, p)
    ss
end

# julia> @b SelectionSampler3(rand(NTuple{6, Float64})) rand
# 2.838 ns

# julia> @b SelectionSampler3(rand(NTuple{6, Float64})) set_weights!(_, rand(NTuple{6, Float64}))
# 10.133 ns

# julia> @b SelectionSampler3(rand(NTuple{64, Float64})) rand
# 19.213 ns

# julia> @b SelectionSampler3(rand(NTuple{64, Float64})) set_weights!(_, rand(NTuple{64, Float64}))
# 86.549 ns

struct SelectionSampler4{N}
    p::MVector{N, Float64}
    o::MVector{N, Int16}
end
function Base.rand(rng::AbstractRNG, ss::SelectionSampler4, lastfull::Int)
    u = rand(rng)*ss.p[lastfull]
    @inbounds for i in 1:lastfull
        ss.p[i] > u && return Int(ss.o[i])
    end
    return lastfull
end
function set_cum_weights!(ss::SelectionSampler4, ns)
    p, lastfull, reorder = ns.sampled_level_weights, length(ns.sampled_levels), ns.reset_order[]
    if reorder > 50*lastfull
        sortperm!(ss.o, p; rev=true)
        ns.reset_order[] = 0
    end
    slevels = ns.sampled_level_numbers
    j = ss.o[1]
    ss.p[1] = p[j]*prec_2pow[slevels[j]+1075]
    @inbounds for i in 2:lastfull
        j = ss.o[i]
        ss.p[i] = ss.p[i-1] + p[j]*prec_2pow[slevels[j]+1075]
    end
    ss
end
function set_level_weights!(ss::SelectionSampler4, ns)
    p, lastfull = ns.sampled_level_weights, length(ns.sampled_levels)
    slevels = ns.sampled_level_numbers
    @inbounds for i in 1:lastfull
        ss.p[i] = p[i]*prec_2pow[slevels[i]+1075]
    end
end

# TODO: add some benchmarks here of SelectionSampler4

struct RejectionSampler3
    length::Base.RefValue{Int}
    data::Vector{Tuple{Int, Float64}}
    maxw::Base.RefValue{Float64}
    RejectionSampler3(i, v) = new(Ref(1), [(i, v)], Ref(v))
end
function Random.rand(rng::AbstractRNG, rs::RejectionSampler3)
    mask = UInt64(1) << Base.top_set_bit(rs.length[] - 1) - 1 # assumes length(data) is the power of two next after (or including) rs.length[]
    maxw = rs.maxw[]
    while true
        u = rand(rng, UInt)
        i = u & mask + 1
        res, x = rs.data[i]
        rand(rng) < x/maxw && return res # TODO: consider reusing random bits from u; a previous test revealed no perf improvement from doing this
    end
end
function Base.push!(rs::RejectionSampler3, i, x)
    len = rs.length[] += 1
    len > length(rs.data) && append!(rs.data, Iterators.repeated((0, UInt64(0)), len-1))
    rs.data[len] = (i, x)
    maxwn = rs.maxw[]
    rs.maxw[] = x > maxwn ? x : maxwn
    rs
end
Base.isempty(rs::RejectionSampler3) = length(rs) == 0 # For testing only
Base.length(rs::RejectionSampler3) = rs.length[] # For testing only

# julia> rs = RejectionSampler3(3, .5)
# RejectionSampler3(Base.RefValue{Int64}(1), [(3, 0.5)])

# julia> @b rs rand
# 14.542 ns

# julia> @b rs push!(_, rand(Int), rand()/2+.5)
# 5.597 ns

# julia> @b rs rand
# 33.750 ns

# julia> rs.length[]
# 1048678

# -----------

struct LinkedListSet3
    data::MVector{34, UInt64}
    LinkedListSet3() = new(zero(MVector{34, UInt64}))
end
Base.in(i::Int, x::LinkedListSet3) = x.data[i >> 6 + 18] & (UInt64(1) << (0x3f - (i & 0x3f))) != 0
Base.push!(x::LinkedListSet3, i::Int) = (x.data[i >> 6 + 18] |= UInt64(1) << (0x3f - (i & 0x3f)); x)
Base.delete!(x::LinkedListSet3, i::Int) = (x.data[i >> 6 + 18] &= ~(UInt64(1) << (0x3f - (i & 0x3f))); x)
function Base.findnext(x::LinkedListSet3, i::Int)
    j = i >> 6 + 18
    k = i & 0x3f
    y = x.data[j] << k
    y != 0 && return i + leading_zeros(y)
    j2 = findnext(!iszero, x.data, j+1)
    j2 === nothing && return nothing
    j2 << 6 + leading_zeros(x.data[j2]) - 18*64
end
function Base.findprev(x::LinkedListSet3, i::Int)
    j = i >> 6 + 18
    k = i & 0x3f
    y = x.data[j] >> (0x3f - k)
    y != 0 && return i - trailing_zeros(y)
    j2 = findprev(!iszero, x.data, j-1)
    j2 === nothing && return nothing
    j2 << 6 - trailing_zeros(x.data[j2]) - 17*64 - 1
end

# ------------------------------

#=
Each entry is assigned a level based on its power.
We have at most min(n, 2048) levels.
# Maintain a distribution over the top N levels and ignore any lower
(or maybe the top log(n) levels and treat the rest as a single level).
For each level, maintain a distribution over the elements of that level
Also, maintain a distribution over the N most significant levels.
To facilitate updating, but unused during sampling, also maintain,
A linked list set (supports push!, delete!, in, findnext, and findprev) of levels
A pointer to the least significant tracked level (-1075 if there are fewer than N levels)
A vector that maps elements (integers) to their level and index in the level

To sample,
draw from the distribution over the top N levels and then
draw from the distribution over the elements of that level.

To add a new element at a given weight,
determine the level of that weight,
create a new level if needed,
add the element to the distribution of that level,
and update the distribution over the top N levels if needed.
Log the location of the new element.

To create a new level,
Push the level into the linked list set of levels.
If the level is below the least significant tracked level, that's all.
Otherwise, update the least significant tracked level and evict an element
from the distribution over the top N levels if necessary

To remove an element,
Lookup the location of the element
Remove the element from the distribution of its level
If the level is now empty, remove the level from the linked list set of levels
If the level is below the least significant tracked level, that's all.
Otherwise, update the least significant tracked level and add an element to the
distribution over the top N levels if possible
=#

if VERSION >= v"1.11.0"
    const prec_2pow = Memory{Float64}([2.0^i for i in -1073:1022])
else
    const prec_2pow = [2.0^i for i in -1073:1022]
end

struct NestedSampler5{N}
    # Used in sampling
    distribution_over_levels::SelectionSampler4{N} # A distribution over 1:N
    sampled_levels::Vector{RejectionSampler3} # The top up to 64 levels TODO: consider making an MVector or SVector

    # Not used in sampling
    sampled_level_weights::MVector{N, Float64} # The weights of the top up to N levels
    sampled_level_numbers::MVector{N, Int} # The level numbers of the top up to N levels TODO: consider merging with sampled_levels_weights or reducing elsize
    all_levels::Vector{Tuple{Double64, RejectionSampler3}} # All the levels, in insertion order, along with their total weights
    level_set::LinkedListSet3 # A set of which levels are present (named by level number)
    level_set_map::Dictionary{Int, Tuple{Int, Int}} # A mapping from level number to index in all_levels and index in sampled_levels (or 0 if not in sampled_levels)
    least_significant_sampled_level::Base.RefValue{Int} # The level number of the least significant tracked level
    entry_info::Vector{Tuple{Int, Int}} # A mapping from element to level number and index in that level (index in level is 0 if entry is not present)
    reset_distribution::Base.RefValue{Bool}
    reset_order::Base.RefValue{Int}
    nvalues::Base.RefValue{Int}
end

NestedSampler5() = NestedSampler5{64}()
NestedSampler5{N}() where N = NestedSampler5{N}(
    SelectionSampler4(zero(MVector{N, Float64}), MVector{N, Int16}(1:N)),
    Tuple{Double64, RejectionSampler3}[],
    zero(MVector{N, Float64}),
    zero(MVector{N, Int}),
    RejectionSampler3[],
    LinkedListSet3(),
    Dictionary{Int, Tuple{Int, Int}}(sizehint=16),
    Ref(-1075),
    Tuple{Int, Int}[],
    Ref(true),
    Ref(0),
    Ref(0)
)

Base.rand(ns::NestedSampler5, n::Integer) = rand(Random.default_rng(), ns, n)
function Base.rand(rng::AbstractRNG, ns::NestedSampler5, n::Integer)
    n < 100 && return [rand(rng, ns) for _ in 1:n]
    slevels = ns.sampled_level_numbers
    lastfull = length(ns.sampled_levels)
    set_level_weights!(ns.distribution_over_levels, ns)
    full_level_weights = @view(ns.distribution_over_levels.p[1:lastfull])
    n_each = rand(rng, Multinomial(n, full_level_weights ./ sum(full_level_weights)))
    inds = Vector{Int}(undef, n)
    q = 1
    @inbounds for (level, k) in enumerate(n_each)
        bucket = ns.sampled_levels[level]
        for _ in 1:k
            inds[q] = rand(rng, bucket)
            q += 1
        end
    end
    shuffle!(rng, inds)
    return inds
end
Base.rand(ns::NestedSampler5) = rand(Random.default_rng(), ns)
@inline function Base.rand(rng::AbstractRNG, ns::NestedSampler5)
    lastfull = length(ns.sampled_levels)
    (ns.reset_distribution[] || ns.reset_order[] > 500*lastfull) && set_cum_weights!(ns.distribution_over_levels, ns)
    ns.reset_distribution[] = false
    level = @inline rand(rng, ns.distribution_over_levels, lastfull)
    @inline rand(rng, ns.sampled_levels[level])
end

@inline function Base.push!(ns::NestedSampler5{N}, i::Int, x::Float64) where N
    ns.reset_distribution[] = true
    ns.nvalues[] += 1
    ns.reset_order[] += 1
    i <= 0 && throw(ArgumentError("Elements must be positive"))
    if i > lastindex(ns.entry_info)
        append!(ns.entry_info, Iterators.repeated((0, 0), i - lastindex(ns.entry_info)))
    elseif ns.entry_info[i] != (0, 0)
        throw(ArgumentError("Element $i is already present"))
    end
    level = exponent(x)
    bucketw = significand(x)/2
    if level ∉ ns.level_set
        # Log the entry
        ns.entry_info[i] = (level, 1)

        # Create a new level (or revive an empty level)
        push!(ns.level_set, level)
        existing_level_indices = get(ns.level_set_map, level, (0, 0))
        all_levels_index = if existing_level_indices == (0, 0)
            level_sampler = RejectionSampler3(i, bucketw)
            push!(ns.all_levels, (Double64(bucketw), level_sampler))
            length(ns.all_levels)
        else
            w, level_sampler = ns.all_levels[existing_level_indices[1]]
            @assert w == 0
            @assert isempty(level_sampler)
            push!(level_sampler, i, bucketw)
            ns.all_levels[existing_level_indices[1]] = (Double64(bucketw), level_sampler)
            existing_level_indices[1]
        end

        # Update the sampled levels if needed
        if level > ns.least_significant_sampled_level[] # we just created a sampled level
            if length(ns.sampled_levels) < N # Add the new level to the top 64
                push!(ns.sampled_levels, level_sampler)
                sl_length = length(ns.sampled_levels)
                ns.sampled_level_weights[sl_length] = bucketw
                ns.sampled_level_numbers[sl_length] = level
                set!(ns.level_set_map, level, (all_levels_index, length(ns.sampled_levels)))
                if length(ns.sampled_levels) == N
                    ns.least_significant_sampled_level[] = findnext(ns.level_set, ns.least_significant_sampled_level[]+1)
                end
            else # Replace the least significant sampled level with the new level
                k, j = ns.level_set_map[ns.least_significant_sampled_level[]]
                ns.level_set_map[ns.least_significant_sampled_level[]] = (k, 0)
                ns.sampled_levels[j] = level_sampler
                ns.sampled_level_weights[j] = bucketw
                ns.sampled_level_numbers[j] = level
                set!(ns.level_set_map, level, (all_levels_index, j))
                ns.least_significant_sampled_level[] = findnext(ns.level_set, ns.least_significant_sampled_level[]+1)
            end
        else # created an unsampled level
            set!(ns.level_set_map, level, (all_levels_index, 0))
        end
    else # Add to an existing level
        j, k = ns.level_set_map[level]
        w, level_sampler = ns.all_levels[j]
        push!(level_sampler, i, bucketw)
        ns.entry_info[i] = (level, length(level_sampler))
        wn = w+Double64(bucketw)
        ns.all_levels[j] = (wn, level_sampler)

        if k != 0 # level is sampled
            ns.sampled_level_weights[k] = Float64(wn)
        end
    end
    ns
end

@inline function Base.delete!(ns::NestedSampler5, i::Int)
    ns.reset_distribution[] = true
    ns.nvalues[] -= 1
    ns.reset_order[] += 1
    if i <= 0 || i > lastindex(ns.entry_info)
        throw(ArgumentError("Element $i is not present"))
    end
    level, j = ns.entry_info[i]
    j == 0 && throw(ArgumentError("Element $i is not present"))
    ns.entry_info[i] = (0, 0)

    l, k = ns.level_set_map[level]
    w, level_sampler = ns.all_levels[l]
    _i, significand = level_sampler.data[j]
    @assert _i == i
    moved_entry, _ = level_sampler.data[j] = level_sampler.data[level_sampler.length[]]
    level_sampler.data[level_sampler.length[]] = (0, UInt64(0))
    level_sampler.length[] -= 1
    if moved_entry != i
        @assert ns.entry_info[moved_entry] == (level, length(level_sampler)+1)
        ns.entry_info[moved_entry] = (level, j)
    end
    wn = w-Double64(significand)
    ns.all_levels[l] = (wn, level_sampler)

    if isempty(level_sampler) # Remove a level
        delete!(ns.level_set, level)
        ns.all_levels[l] = (Double64(0), level_sampler) # Fixup for rounding error
        if k != 0 # Remove a sampled level
            replacement = findprev(ns.level_set, ns.least_significant_sampled_level[]-1)
            ns.level_set_map[level] = (l, 0)
            if replacement === nothing # We'll now have fewer than N sampled levels
                ns.least_significant_sampled_level[] = -1075
                sl_length = length(ns.sampled_levels)
                moved_level = ns.sampled_level_numbers[sl_length]
                if moved_level == level
                    pop!(ns.sampled_levels)
                    ns.sampled_level_weights[sl_length] = 0.0
                    # sampled_level_numbers can have unclean trailing values
                else
                    ns.sampled_level_numbers[k] = moved_level
                    ns.sampled_levels[k] = pop!(ns.sampled_levels)
                    ns.sampled_level_weights[k] = ns.sampled_level_weights[sl_length]
                    ns.sampled_level_weights[sl_length] = 0.0
                    all_index, _length_sampled_levels_plus_one = ns.level_set_map[moved_level]
                    @assert _length_sampled_levels_plus_one == length(ns.sampled_levels)+1
                    ns.level_set_map[moved_level] = (all_index, k)
                end
            else # Replace the removed level with the replacement
                ns.least_significant_sampled_level[] = replacement
                all_index, _zero = ns.level_set_map[replacement]
                @assert _zero == 0
                ns.level_set_map[replacement] = (all_index, k)
                w, replacement_level = ns.all_levels[all_index]
                ns.sampled_levels[k] = replacement_level
                ns.sampled_level_weights[k] = Float64(w)
                ns.sampled_level_numbers[k] = replacement
            end
        end
    elseif k != 0
        ns.sampled_level_weights[k] = Float64(wn)
    end

    ns
end

Base.isempty(ns::NestedSampler5) = ns.nvalues[] == 0

struct SamplerIndices{I}
    ns::NestedSampler5
    iter::I
end
function SamplerIndices(ns::NestedSampler5)
    iter = Iterators.Filter(x -> x != 0, Iterators.Flatten((Iterators.map(x -> x[1], b[2].data) for b in ns.all_levels)))
    SamplerIndices(ns, iter)
end
Base.iterate(inds::SamplerIndices) = Base.iterate(inds.iter)
Base.iterate(inds::SamplerIndices, state) = Base.iterate(inds.iter, state)
Base.eltype(::Type{<:SamplerIndices}) = Int
Base.IteratorSize(::Type{<:SamplerIndices}) = Base.HasLength()
Base.length(inds::SamplerIndices) = inds.ns.nvalues[]

const DynamicDiscreteSampler = NestedSampler5

# ------------------------------

# Trash:

struct RejectionSampler
    maximum::Float64
    p::Vector{Tuple{Int, Float64}}
end
function set_weight!(rs::RejectionSampler, i, p)
    rs.p = [(i, p[i]) for i in 1:length(p)]
    rs.maximum[] = maximum(p)
    rs
end

struct Reducer0{T}
    group_p::T
    groups::Vector{Vector{Tuple{Int, Float64}}}
    small::Vector{Tuple{Int, Float64}}
    p::Ref{Float64}
    metadata::Vector{@NamedTuple{group::Int, group_index::Int, weight::Float64}} # For mutation only
end
function sample_from_group(group::Vector{Tuple{Int, Float64}})
    while true
        u = rand(UInt)
        i = u & (length(group)-1) + 1 # assumes length(group) is a power of two
        res,p = group[i]
        u > p || return res
    end
end
function Random.rand(r::Reducer0)
    group = if rand() < r.p[]
        (r.small)
    else
        groups[rand(r.group_p)]
    end
    sample_from_group(group)
end

roundup(x) = 1 << Base.top_set_bit(x-1)
function Reducer0(group_p_maker, weights)
    n = roundup(length(weights))
    log2(w)
end

end
