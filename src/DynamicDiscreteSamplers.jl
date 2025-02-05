module DynamicDiscreteSamplers

export DynamicDiscreteSampler, SamplerIndices

using Distributions, Random, StaticArrays

struct SelectionSampler{N}
    p::MVector{N, Float64}
    o::MVector{N, Int16}
end
function Base.rand(rng::AbstractRNG, ss::SelectionSampler, lastfull::Int)
    u = rand(rng)*ss.p[lastfull]
    @inbounds for i in lastfull-1:-1:1
        ss.p[i] < u && return i+1
    end
    return 1
end
function set_cum_weights!(ss::SelectionSampler, ns, reorder)
    p, lastfull = ns.sampled_level_weights, ns.track_info.lastfull
    if reorder
        ns.track_info.reset_order = 0
        if !ns.track_info.reset_distribution && issorted(@view(p[1:lastfull]))
            return ss
        end
        @inline reorder_levels(ns, ss, p, lastfull)
        ns.track_info.firstchanged = 1
    end
    firstc = ns.track_info.firstchanged
    ss.p[1] = p[1]
    f = firstc + Int(firstc == 1)
    @inbounds for i in f:lastfull
        ss.p[i] = ss.p[i-1] + p[i]
    end
    ns.track_info.firstchanged = lastfull
    return ss
end
function reorder_levels(ns, ss, p, lastfull)
    sortperm!(@view(ss.o[1:lastfull]), @view(p[1:lastfull]); alg=Base.Sort.InsertionSortAlg())
    @inbounds for i in 1:lastfull
        if ss.o[i] == zero(Int16)
            all_index = ns.level_set_map.indices[ns.sampled_level_numbers[i]+1075][1]
            ns.level_set_map.indices[ns.sampled_level_numbers[i]+1075] = (all_index, i)
            continue
        end
        value1 = ns.sampled_levels[i]
        value2 = ns.sampled_level_numbers[i]
        value3 = p[i]
        x, y = i, Int(ss.o[i])
        while y != i
            ss.o[x] = zero(Int16)
            ns.sampled_levels[x] = ns.sampled_levels[y]
            ns.sampled_level_numbers[x] = ns.sampled_level_numbers[y]
            p[x] = p[y]
            x = y
            y = Int(ss.o[x])
        end
        ns.sampled_levels[x] = value1
        ns.sampled_level_numbers[x] = value2
        p[x] = value3
        ss.o[x] = zero(Int16)
        all_index = ns.level_set_map.indices[ns.sampled_level_numbers[i]+1075][1]
        ns.level_set_map.indices[ns.sampled_level_numbers[i]+1075] = (all_index, i)
    end
end

mutable struct RejectionInfo
    length::Int
    maxw::Float64
    mask::UInt64
end
struct RejectionSampler
    data::Vector{Tuple{Int, Float64}}
    track_info::RejectionInfo
    RejectionSampler(i, v) = new([(i, v)], RejectionInfo(1, v, zero(UInt)))
end
function Random.rand(rng::AbstractRNG, rs::RejectionSampler, f::Function)
    len = rs.track_info.length
    mask = rs.track_info.mask
    maxw = rs.track_info.maxw
    while true
        u = rand(rng, UInt64)
        i = Int(u & mask)
        i >= len && continue
        i += 1
        res, x = rs.data[i]
        f(rng, u) * maxw < x && return (i, res)
    end
end
@inline randreuse(rng, u) = Float64(u >>> 11) * 0x1.0p-53
@inline randnoreuse(rng, _) = rand(rng)
function Base.push!(rs::RejectionSampler, i, x)
    len = rs.track_info.length += 1
    if len > length(rs.data)
        resize!(rs.data, 2*length(rs.data))
        rs.track_info.mask = UInt(1) << (8*sizeof(len-1) - leading_zeros(len-1)) - 1
    end
    rs.data[len] = (i, x)
    maxwn = rs.track_info.maxw
    rs.track_info.maxw = ifelse(x > maxwn, x, maxwn)
    rs
end
Base.isempty(rs::RejectionSampler) = length(rs) == 0 # For testing only
Base.length(rs::RejectionSampler) = rs.track_info.length # For testing only

struct LinkedListSet
    data::SizedVector{34, UInt64, Vector{UInt64}}
    LinkedListSet() = new(zeros(UInt64, 34))
end
Base.in(i::Int, x::LinkedListSet) = x.data[i >> 6 + 18] & (UInt64(1) << (0x3f - (i & 0x3f))) != 0
Base.push!(x::LinkedListSet, i::Int) = (x.data[i >> 6 + 18] |= UInt64(1) << (0x3f - (i & 0x3f)); x)
Base.delete!(x::LinkedListSet, i::Int) = (x.data[i >> 6 + 18] &= ~(UInt64(1) << (0x3f - (i & 0x3f))); x)
function Base.findnext(x::LinkedListSet, i::Int)
    j = i >> 6 + 18
    @inbounds y = x.data[j] << (i & 0x3f)
    y != 0 && return i + leading_zeros(y)
    for j2 in j+1:34
        @inbounds c = x.data[j2]
        !iszero(c) && return j2 << 6 + leading_zeros(c) - 18*64
    end
    return -10000
end
function Base.findprev(x::LinkedListSet, i::Int)
    j = i >> 6 + 18
    @inbounds y = x.data[j] >> (0x3f - i & 0x3f)
    y != 0 && return i - trailing_zeros(y)
    for j2 in j-1:-1:1
        @inbounds c = x.data[j2]
        !iszero(c) && return j2 << 6 - trailing_zeros(c) - 17*64 - 1
    end
    return -10000
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

struct LevelMap
    presence::BitVector
    indices::Vector{Tuple{Int, Int}}
    function LevelMap()
        presence = BitVector()
        resize!(presence, 2098)
        fill!(presence, false)
        indices = Vector{Tuple{Int, Int}}(undef, 2098)
        return new(presence, indices)
    end
end

struct EntryInfo
    presence::BitVector
    indices::Vector{Int}
    EntryInfo() = new(BitVector(), Int[])
end

mutable struct TrackInfo
    lastsampled_idx::Int
    lastsampled_idx_out::Int
    lastsampled_idx_in::Int
    least_significant_sampled_level::Int # The level number of the least significant tracked level
    nvalues::Int
    firstchanged::Int
    lastfull::Int
    reset_order::Int
    reset_distribution::Bool
end

@inline sig(x::Float64) = (reinterpret(UInt64, x) & Base.significand_mask(Float64)) + Base.significand_mask(Float64) + 1

@inline function flot(sg::UInt128, level::Integer)
    shift = Int64(8 * sizeof(sg) - 53 - leading_zeros(sg))
    x = (sg >>= shift) % UInt64
    exp = level + shift + 1022
    reinterpret(Float64, x + (exp << 52))
end

struct NestedSampler{N}
    # Used in sampling
    distribution_over_levels::SelectionSampler{N} # A distribution over 1:N
    sampled_levels::MVector{N, Int16} # The top up to N levels indices
    all_levels::Vector{Tuple{UInt128, RejectionSampler}} # All the levels, in insertion order, along with their total weights

    # Not used in sampling
    sampled_level_weights::MVector{N, Float64} # The weights of the top up to N levels
    sampled_level_numbers::MVector{N, Int16} # The level numbers of the top up to N levels
    level_set::LinkedListSet # A set of which levels are non-empty (named by level number)
    level_set_map::LevelMap # A mapping from level number to index in all_levels and index in sampled_levels (or 0 if not in sampled_levels)
    entry_info::EntryInfo # A mapping from element to level number and index in that level (index in level is 0 if entry is not present)
    track_info::TrackInfo
end

NestedSampler() = NestedSampler{64}()
NestedSampler{N}() where N = NestedSampler{N}(
    SelectionSampler(zero(MVector{N, Float64}), MVector{N, Int16}(1:N)),
    zero(MVector{N, Int16}),
    Tuple{UInt128, RejectionSampler}[],
    zero(MVector{N, Float64}),
    zero(MVector{N, Int16}),
    LinkedListSet(),
    LevelMap(),
    EntryInfo(),
    TrackInfo(0, 0, 0, -1075, 0, 1, 0, 0, true),
)

Base.rand(ns::NestedSampler, n::Integer) = rand(Random.default_rng(), ns, n)
function Base.rand(rng::AbstractRNG, ns::NestedSampler, n::Integer)
    n < 100 && return [rand(rng, ns) for _ in 1:n]
    lastfull = ns.track_info.lastfull
    ws = @view(ns.sampled_level_weights[1:lastfull])
    totw = sum(ws)
    maxw = maximum(ws)
    maxw/totw > 0.98 && return [rand(rng, ns) for _ in 1:n]
    n_each = rand(rng, Multinomial(n, ws ./ totw))
    inds = Vector{Int}(undef, n)
    q = 1
    @inbounds for (level, k) in enumerate(n_each)
        bucket = ns.all_levels[Int(ns.sampled_levels[level])][2]
        f = length(bucket) <= 2048 ? randreuse : randnoreuse
        for _ in 1:k
            ti = @inline rand(rng, bucket, f)
            inds[q] = ti[2]
            q += 1
        end
    end
    shuffle!(rng, inds)
    return inds
end
Base.rand(ns::NestedSampler) = rand(Random.default_rng(), ns)
@inline function Base.rand(rng::AbstractRNG, ns::NestedSampler)
    track_info = ns.track_info
    track_info.reset_order += 1
    lastfull = track_info.lastfull
    reorder = lastfull > 8 && track_info.reset_order > 300*lastfull
    if track_info.reset_distribution || reorder
        @inline set_cum_weights!(ns.distribution_over_levels, ns, reorder)
        track_info.reset_distribution = false
    end
    level = @inline rand(rng, ns.distribution_over_levels, lastfull)
    j, i = @inline rand(rng, ns.all_levels[Int(ns.sampled_levels[level])][2], randnoreuse)
    track_info.lastsampled_idx = i
    track_info.lastsampled_idx_out = level
    track_info.lastsampled_idx_in = j
    return i
end

function Base.append!(ns::NestedSampler{N}, inds::Union{AbstractRange{Int}, Vector{Int}},
        xs::Union{AbstractRange{Float64}, Vector{Float64}}) where N
    ns.track_info.reset_distribution = true
    ns.track_info.reset_order += length(inds)
    ns.track_info.nvalues += length(inds)
    maxi = maximum(inds)
    l_info = lastindex(ns.entry_info.presence)
    if maxi > l_info
        newl = max(2*l_info, maxi)
        resize!(ns.entry_info.indices, newl)
        resize!(ns.entry_info.presence, newl)
        fill!(@view(ns.entry_info.presence[l_info+1:newl]), false)
    end
    for (i, x) in zip(inds, xs)
        if ns.entry_info.presence[i]
            throw(ArgumentError("Element $i is already present"))
        end
        _push!(ns, i, x)
    end
    return ns
end

@inline function Base.push!(ns::NestedSampler{N}, i::Int, x::Float64) where N
    ns.track_info.reset_distribution = true
    ns.track_info.reset_order += 1
    ns.track_info.nvalues += 1
    i <= 0 && throw(ArgumentError("Elements must be positive"))
    x <= 0.0 && throw(ArgumentError("Weights must be positive"))
    l_info = lastindex(ns.entry_info.presence)
    if i > l_info
        newl = max(2*l_info, i)
        resize!(ns.entry_info.indices, newl)
        resize!(ns.entry_info.presence, newl)
        fill!(@view(ns.entry_info.presence[l_info+1:newl]), false)
    elseif ns.entry_info.presence[i]
        throw(ArgumentError("Element $i is already present"))
    end
    return _push!(ns, i, x)
end

@inline function _push!(ns::NestedSampler{N}, i::Int, x::Float64) where N
    bucketw, level = frexp(x)
    level -= 1
    level_b16 = Int16(level)
    ns.entry_info.presence[i] = true
    if level ∉ ns.level_set
        # Log the entry
        ns.entry_info.indices[i] = 4096 + level + 1075

        # Create a new level (or revive an empty level)
        push!(ns.level_set, level)
        existing_level_indices = ns.level_set_map.presence[level+1075]
        all_levels_index = if !existing_level_indices
            level_sampler = RejectionSampler(i, bucketw)
            push!(ns.all_levels, (sig(x), level_sampler))
            length(ns.all_levels)
        else
            level_indices = ns.level_set_map.indices[level+1075]
            w, level_sampler = ns.all_levels[level_indices[1]]
            @assert w == 0
            @assert isempty(level_sampler)
            push!(level_sampler, i, bucketw)
            ns.all_levels[level_indices[1]] = (sig(x), level_sampler)
            level_indices[1]
        end
        ns.level_set_map.presence[level+1075] = true

        # Update the sampled levels if needed
        if level > ns.track_info.least_significant_sampled_level # we just created a sampled level
            if ns.track_info.lastfull < N # Add the new level to the top 64
                ns.track_info.lastfull += 1
                sl_length = ns.track_info.lastfull
                ns.sampled_levels[sl_length] = Int16(all_levels_index)
                ns.sampled_level_weights[sl_length] = x
                ns.sampled_level_numbers[sl_length] = level_b16
                ns.level_set_map.indices[level+1075] = (all_levels_index, sl_length)
                if sl_length == N
                    ns.track_info.least_significant_sampled_level = findnext(ns.level_set, ns.track_info.least_significant_sampled_level+1)
                end
            else # Replace the least significant sampled level with the new level
                j, k = ns.level_set_map.indices[ns.track_info.least_significant_sampled_level+1075]
                ns.level_set_map.indices[ns.track_info.least_significant_sampled_level+1075] = (j, 0)
                ns.sampled_levels[k] = Int16(all_levels_index)
                ns.sampled_level_weights[k] = x
                ns.sampled_level_numbers[k] = level_b16
                ns.level_set_map.indices[level+1075] = (all_levels_index, k)
                ns.track_info.least_significant_sampled_level = findnext(ns.level_set, ns.track_info.least_significant_sampled_level+1)
                firstc = ns.track_info.firstchanged
                ns.track_info.firstchanged = ifelse(k < firstc, k, firstc)
            end
        else # created an unsampled level
            ns.level_set_map.indices[level+1075] = (all_levels_index, 0)
        end
    else # Add to an existing level
        j, k = ns.level_set_map.indices[level+1075]
        w, level_sampler = ns.all_levels[j]
        push!(level_sampler, i, bucketw)
        ns.entry_info.indices[i] = length(level_sampler) << 12 + level + 1075
        wn = w+sig(x)
        ns.all_levels[j] = (wn, level_sampler)

        if k != 0 # level is sampled
            ns.sampled_level_weights[k] = flot(wn, level)
            firstc = ns.track_info.firstchanged
            ns.track_info.firstchanged = ifelse(k < firstc, k, firstc)
        end
    end
    return ns
end

@inline function Base.delete!(ns::NestedSampler, i::Int)
    ns_track_info = ns.track_info
    ns_track_info.reset_distribution = true
    ns_track_info.reset_order += 1
    ns_track_info.nvalues -= 1
    if i <= 0 || i > lastindex(ns.entry_info.presence)
        throw(ArgumentError("Element $i is not present"))
    end
    if ns_track_info.lastsampled_idx == i
        level = Int(ns.sampled_level_numbers[ns_track_info.lastsampled_idx_out])
        j = ns_track_info.lastsampled_idx_in
    else
        c = ns.entry_info.indices[i]
        level = c & 4095 - 1075
        j = (c - level - 1075) >> 12
    end
    ns_track_info.lastsampled_idx = 0
    !ns.entry_info.presence[i] && throw(ArgumentError("Element $i is not present"))
    ns.entry_info.presence[i] = false

    l, k = ns.level_set_map.indices[level+1075]
    w, level_sampler = ns.all_levels[l]
    _i, significand = level_sampler.data[j]
    @assert _i == i
    len = level_sampler.track_info.length
    moved_entry, _ = level_sampler.data[j] = level_sampler.data[len]
    level_sampler.track_info.length -= 1
    if (len & (len-1)) == 0
        level_sampler.track_info.mask = UInt(1) << (8*sizeof(len-1) - leading_zeros(len-1)) - 1
    end
    if moved_entry != i
        @assert ns.entry_info.indices[moved_entry] == (length(level_sampler)+1) << 12 + level + 1075
        ns.entry_info.indices[moved_entry] = j << 12 + level + 1075
    end
    wn = w-sig(significand*exp2(level+1))
    ns.all_levels[l] = (wn, level_sampler)

    if isempty(level_sampler) # Remove a level
        delete!(ns.level_set, level)
        if k != 0 # Remove a sampled level
            firstc = ns.track_info.firstchanged
            ns.track_info.firstchanged = ifelse(k < firstc, k, firstc)
            replacement = findprev(ns.level_set, ns_track_info.least_significant_sampled_level-1)
            ns.level_set_map.indices[level+1075] = (l, 0)
            if replacement == -10000 # We'll now have fewer than N sampled levels
                ns_track_info.least_significant_sampled_level = -1075
                sl_length = ns_track_info.lastfull
                ns_track_info.lastfull -= 1
                moved_level = ns.sampled_level_numbers[sl_length]
                if moved_level == Int16(level)
                    ns.sampled_level_weights[sl_length] = 0.0
                else
                    ns.sampled_level_numbers[k], ns.sampled_level_numbers[sl_length] = ns.sampled_level_numbers[sl_length], ns.sampled_level_numbers[k]
                    ns.sampled_levels[k], ns.sampled_levels[sl_length] = ns.sampled_levels[sl_length], ns.sampled_levels[k]
                    ns.sampled_level_weights[k] = ns.sampled_level_weights[sl_length]
                    ns.sampled_level_weights[sl_length] = 0.0
                    all_index, _l = ns.level_set_map.indices[ns.sampled_level_numbers[k]+1075]
                    @assert _l == ns.track_info.lastfull+1
                    ns.level_set_map.indices[ns.sampled_level_numbers[k]+1075] = (all_index, k)
                    all_index = ns.level_set_map.indices[ns.sampled_level_numbers[sl_length]+1075][1]
                    ns.level_set_map.indices[ns.sampled_level_numbers[sl_length]+1075] = (all_index, sl_length)
                end
            else # Replace the removed level with the replacement
                ns_track_info.least_significant_sampled_level = replacement
                all_index, _zero = ns.level_set_map.indices[replacement+1075]
                @assert _zero == 0
                ns.level_set_map.indices[replacement+1075] = (all_index, k)
                w, replacement_level = ns.all_levels[all_index]
                ns.sampled_levels[k] = Int16(all_index)
                ns.sampled_level_weights[k] = flot(w, replacement)
                ns.sampled_level_numbers[k] = replacement
            end
        end
    elseif k != 0
        ns.sampled_level_weights[k] = flot(wn, level)
        firstc = ns.track_info.firstchanged
        ns.track_info.firstchanged = ifelse(k < firstc, k, firstc)
    end
    return ns
end

Base.in(i::Int, ns::NestedSampler) = 0 < i <= length(ns.entry_info.presence) && ns.entry_info.presence[i]

Base.isempty(ns::NestedSampler) = ns.track_info.nvalues == 0

struct SamplerIndices{I}
    ns::NestedSampler
    iter::I
end
function SamplerIndices(ns::NestedSampler)
    iter = Iterators.Flatten((Iterators.map(x -> x[1], @view(b[2].data[1:b[2].track_info.length])) for b in ns.all_levels))
    SamplerIndices(ns, iter)
end
Base.iterate(inds::SamplerIndices) = Base.iterate(inds.iter)
Base.iterate(inds::SamplerIndices, state) = Base.iterate(inds.iter, state)
Base.eltype(::Type{<:SamplerIndices}) = Int
Base.IteratorSize(::Type{<:SamplerIndices}) = Base.HasLength()
Base.length(inds::SamplerIndices) = inds.ns.track_info.nvalues

# const DynamicDiscreteSampler = NestedSampler

isdefined(@__MODULE__, :Memory) || const Memory = Vector # Compat for Julia < 1.11

const DEBUG = Base.JLOptions().check_bounds == 1
_convert(T, x) = DEBUG ? T(x) : x%T
macro check(arg)
    DEBUG ? esc(:(@assert $arg)) : nothing
end

# Take Two!
# - Exact
# - O(1) in theory
# - Fast in practice

#=

levels are powers of two. Each level has a true weight which is the sum of the (Float64) weights of
the elements in that level and is represented as a UInt128 which is the sum of the significands of that level (exponent stored implicitly).
Each level also has a an approximate weight which is represented as a UInt64 with an implicit "<< level0" where
level0 is a constant maintained by the sampler so that the sum of the approximate weights is less
than 2^64 and greator than 2^32. The sum of the approximate weights and index of the highest level
are also maintained.

To select a level, pick a random number in Base.OneTo(sum of approximate weights) and find that
level with linear search

maintain the total weight

=#

abstract type Weights <: AbstractVector{Float64} end
struct FixedSizeWeights <: Weights
    m::Memory{UInt64}
    global _FixedSizeWeights
    _FixedSizeWeights(m::Memory{UInt64}) = new(m)
end
struct SemiResizableWeights <: Weights
    m::Memory{UInt64}
    SemiResizableWeights(w::FixedSizeWeights) = new(w.m)
end
mutable struct ResizableWeights <: Weights
    m::Memory{UInt64}
    ResizableWeights(w::FixedSizeWeights) = new(w.m)
end

## Standard memory layout: (TODO: add alternative layout for small cases)

# <memory_length::Int>
# 1                      length::Int
# 2                      max_level::Int # absolute pointer to the first element of level weights that is nonzero
# 3                      shift::Int level weights are euqal to shifted_significand_sums<<(exponent_bits+shift) rounded up
# 4                      sum(level weights)::UInt64
# 5..2050                level weights::[UInt64 2046] # earlier is higher. first is exponent bits 0x7fe, last is exponent bits 0x001. Subnormal are not supported.
# 2051..6142             shifted_significand_sums::[UInt128 2046] # sum of significands shifted by 11 bits to the left with their leading 1s appended (the maximum significand contributes 0xfffffffffffff800)
# 6143..10234            level location info::[NamedTuple{posm2::Int, length::Int} 2046] indexes into sub_weights, posm2 is absolute into m.

# gc info:
# 10235                  next_free_space::Int (used to re-allocate) <index 10235>
# 16 unused bits
# 10236..10491           level allocated length::[UInt8 2046] (2^(x-1) is implied)

# 10492..10491+2len      edit_map (maps index to current location in sub_weights)::[pos::Int, exponent::Int] (zero means zero; fixed location, always at the start. Force full realloc when it OOMs. exponent could be UInt11, lots of wasted bits)

# 10492+2allocated_len..10491+2allocated_len+6len sub_weights (woven with targets)::[[shifted_significand::UInt64, target::Int}]]. allocated_len == length_from_memory(length(m))

# shifted significands are stored in sub_weights with their implicit leading 1 adding and shifted left 11 bits
#     element_from_sub_weights = 0x8000000000000000 | (reinterpret(UInt64, weight::Float64) << 11)
# And sampled with
#     rand(UInt64) < element_from_sub_weights
# this means that for the lowest normal significand (52 zeros with an implicit leading one),
# achieved by 2.0, 4.0, etc the shifted significand stored in sub_weights is 0x8000000000000000
# and there are 2^63 pips less than that value (1/2 probability). For the
# highest normal significand (52 ones with an implicit leading 1) the shifted significand
# stored in sub_weights is 0xfffffffffffff800 and there are 2^64-2^11 pips less than
# that value for a probability of (2^64-2^11) / 2^64 == (2^53-1) / 2^53 == prevfloat(2.0)/2.0
@check 0xfffffffffffff800//big(2)^64 == (UInt64(2)^53-1)//UInt64(2)^53 == big(prevfloat(2.0))/big(2.0)
@check 0x8000000000000000 | (reinterpret(UInt64, 1.0::Float64) << 11) === 0x8000000000000000
@check 0x8000000000000000 | (reinterpret(UInt64, prevfloat(1.0)::Float64) << 11) === 0xfffffffffffff800
# shifted significand sums are literal sums of the element_from_sub_weights's (though stored
# as UInt128s because any two element_from_sub_weights's will overflow when added).

# target can also store metadata useful for compaction.
# the range 0x0000000000000001 to 0x7fffffffffffffff (1:typemax(Int)) represents literal targets
# the range 0x8000000000000001 to 0x80000000000007fe indicates that this is an empty but non-abandoned group with exponent bits target-0x8000000000000000
# the range 0xc000000000000000 to 0xffffffffffffffff indicates that the group is abandoned and has length -target.

## Initial API:

# setindex!, getindex, resize! (auto-zeros), scalar rand
# Trivial extensions:
# push!, delete!

# TODO for performance and simplicity, change weights to rounded down instead of rounded up
# then, leverage the fact that plain bitshifting rounds down by default.

Base.rand(rng::AbstractRNG, w::Weights) = _rand(rng, w.m)
Base.getindex(w::Weights, i::Int) = _getindex(w.m, i)
Base.setindex!(w::Weights, v, i::Int) = (_setindex!(w.m, Float64(v), i); w)

#=@inbounds=# function _rand(rng::AbstractRNG, m::Memory{UInt64})

    @label reject

    # Select level
    x = rand(rng, Base.OneTo(m[4]))
    i = m[2]
    local mi
    while #=i < 2046+4=# true
        mi = m[i]
        x <= mi && break
        x -= mi
        i += 1
    end

    # Low-probability rejection to improve accuracy from very close to perfect
    if x == mi # mi is the weight rounded up. If they are equal than we should refine futher and possibly reject. This branch is very uncommon and still O(1); constant factors don't matter here.
        # significand_sum::UInt128 = ...
        # weight::UInt64 = mi = ceil(significand_sum<<*(exponent_bits+shift))...
        # rejection_p = ceil(significand_sum<<*(exponent_bits+shift)) - true(ceil(significand_sum<<*(exponent_bits+shift)))
        # rejection_p = 1-rem(significand_sum<<*(exponent_bits+shift))
        # acceptance_p = rem(significand_sum<<*(exponent_bits+shift))
        # acceptance_p = significand_sum<<*(exponent_bits+shift) & ...00000.111111...
        j = 2i+2041
        exponent_bits = 0x7fe+5-i
        shift = signed(exponent_bits + m[3])
        significand_sum = merge_uint64(m[j], m[j+1])
        while true
            x = rand(rng, UInt64)
            # p_stage = significand_sum << shift & ...00000.111111...64...11110000
            target = significand_sum << (shift + 64) % UInt64
            x > target && @goto reject
            x < target && break
            shift += 64
            shift >= 0 && break
        end
    end

    # Lookup level info
    j = 2i + 6133
    posm2 = m[j]
    len = m[j+1]

    # Sample within level
    while true
        r = rand(rng, UInt64)
        k1 = (r>>leading_zeros(len-1))
        k2 = k1<<1+posm2+2 # TODO for perf: try %Int here (and everywhere)
        # TODO for perf: delete the k1 < len check by maintaining all the out of bounds m[k2] equal to 0
        k1 < len && rand(rng, UInt64) < m[k2] && return Int(signed(m[k2+1]))
    end
end

function _getindex(m::Memory{UInt64}, i::Int)
    @boundscheck 1 <= i <= m[1] || throw(BoundsError(_FixedSizeWeights(m), i))
    j = 2i + 10490
    pos = m[j]
    pos == 0 && return 0.0
    exponent = m[j+1]
    weight = m[pos]
    reinterpret(Float64, exponent | (weight - 0x8000000000000000) >> 11)
end

function _setindex!(m::Memory, v::Float64, i::Int)
    @boundscheck 1 <= i <= m[1] || throw(BoundsError(_FixedSizeWeights(m), i))
    uv = reinterpret(UInt64, v)
    if uv == 0
        _set_to_zero!(m, i)
        return
    end
    0x0010000000000000 <= uv <= 0x7fefffffffffffff || throw(DomainError(v, "Invalid weight")) # Excludes subnormals

    # Find the entry's pos in the edit map table
    j = 2i + 10490
    pos = m[j]
    if pos == 0
        _set_from_zero!(m, v, i::Int)
    else
        _set_nonzero!(m, v, i::Int)
    end
end

function _set_nonzero!(m, v, i)
    # TODO for performance: join these two opperations
    _set_to_zero!(m, i)
    _set_from_zero!(m, v, i)
end

function _set_from_zero!(m::Memory, v::Float64, i::Int)
    uv = reinterpret(UInt64, v)
    j = 2i + 10490
    @check m[j] == 0

    exponent = uv & Base.exponent_mask(Float64)
    m[j+1] = exponent
    # update group total weight and total weight
    shifted_significand_sum_index = get_shifted_significand_sum_index(exponent)
    shifted_significand_sum = get_UInt128(m, shifted_significand_sum_index)
    shifted_significand = 0x8000000000000000 | uv << 11
    shifted_significand_sum += shifted_significand
    set_UInt128!(m, shifted_significand_sum, shifted_significand_sum_index)
    weight_index = 5 + 0x7fe - exponent >> 52
    if m[4] == 0 # if we were empty, set global shift (m[3]) so that m[4] will become ~2^40.
        m[3] = -24 - exponent >> 52
        weight = compute_weight(m, exponent, shifted_significand_sum) # TODO for perf: inline
        @check Base.top_set_bit(weight) == 40 # TODO for perf: delete
        m[weight_index] = weight
        m[4] = weight
    else
        update_weights!(m, exponent, shifted_significand_sum) # TODO for perf: inline
    end
    m[2] = min(m[2], weight_index) # Set after insertion because update_weights! may need to update the global shift, in which case knowing the old m[2] will help it skip checking empty levels

    # lookup the group by exponent and bump length
    group_length_index = shifted_significand_sum_index + 2*2046 + 1
    group_posm2 = m[group_length_index-1]
    group_length = m[group_length_index]+1
    m[group_length_index] = group_length # setting this before compaction means that compaction will ensure there is enough space for this expanded group, but will also copy one index (16 bytes) of junk which could access past the end of m. The junk isn't an issue once coppied because we immediately overwrite it. The former (copying past the end of m) only happens if the group to be expanded is already kissing the end. In this case, it will end up at the end after compaction and be easily expanded afterwords. Consequently, we treat that case specially and bump group length and manually expand after compaction
    allocs_index,allocs_subindex = get_alloced_indices(exponent)
    allocs_chunk = m[allocs_index]
    log2_allocated_size = allocs_chunk >> allocs_subindex % UInt8 - 1
    allocated_size = 1<<log2_allocated_size

    # if there is not room in the group, shift and expand
    if group_length > allocated_size
        next_free_space = m[10235]
        # if at end already, simply extend the allocation # TODO see if removing this optimization is problematic; TODO verify the optimization is triggering
        if next_free_space == group_posm2+2group_length # note that this is valid even if group_length is 1 (previously zero).
            new_allocation_length = max(2, 2allocated_size)
            new_next_free_space = next_free_space+new_allocation_length
            if new_next_free_space > length(m)+1 # There isn't room; we need to compact
                m[group_length_index] = group_length-1 # See comment above; we don't want to copy past the end of m
                firstindex_of_compactee = 2length_from_memory(length(m)) + 10492 # TODO for clarity: move this into compact!
                next_free_space = compact!(m, Int(firstindex_of_compactee), m, Int(firstindex_of_compactee))
                group_posm2 = next_free_space-new_allocation_length-2 # The group will move but remian the last group
                new_next_free_space = next_free_space+new_allocation_length
                @check new_next_free_space < length(m)+1 # TODO for perf, delete this
                m[group_length_index] = group_length

                # Re-lookup allocated chunk because compaction could have changed other
                # chunk elements. However, the allocated size of this group could not have
                # changed because it was previously maxed out.
                allocs_chunk = m[allocs_index]
                @check log2_allocated_size == allocs_chunk >> allocs_subindex % UInt8 - 1
                @check allocated_size == 1<<log2_allocated_size
            end
            # expand the allocated size and bump next_free_space
            new_chunk = allocs_chunk + UInt64(1) << allocs_subindex
            m[allocs_index] = new_chunk
            m[10235] = new_next_free_space
        else # move and reallocate (this branch also handles creating new groups: TODO expirment with perf and clarity by splicing that branch out)
            twice_new_allocated_size = max(0x2,allocated_size<<2)
            new_next_free_space = next_free_space+twice_new_allocated_size
            if new_next_free_space > length(m)+1 # out of space; compact. TODO for perf, consider resizing at this time slightly eagerly?
                firstindex_of_compactee = 2length_from_memory(length(m)) + 10492
                m[group_length_index] = group_length-1 # incrementing the group length before compaction is spotty because if the group was previously empty then this new group length will be ignored (compact! loops over sub_weights, not levels)
                next_free_space = compact!(m, Int(firstindex_of_compactee), m, Int(firstindex_of_compactee))
                m[group_length_index] = group_length
                new_next_free_space = next_free_space+twice_new_allocated_size
                @check new_next_free_space < length(m)+1 # After compaction there should be room TODO for perf, delete this

                group_posm2 = m[group_length_index-1] # The group likely moved during compaction

                # Re-lookup allocated chunk because compaction could have changed other
                # chunk elements. However, the allocated size of this group could not have
                # changed because it was previously maxed out.
                allocs_chunk = m[allocs_index]
                @check log2_allocated_size == allocs_chunk >> allocs_subindex % UInt8 - 1
                @check allocated_size == 1<<log2_allocated_size
            end
            # TODO for perf: make compact! re-allocate the expanded group larger so there's no need to double the allocated size here if the compact branch is taken
            # TODO for perf, try removing the moveie before compaction (tricky: where to store that info?)
            # TODO make this whole alg dry, but only after setting up robust benchmarks in CI

            # expand the allocated size and bump next_free_space
            new_chunk = allocs_chunk + UInt64(1) << allocs_subindex
            m[allocs_index] = new_chunk

            m[10235] = new_next_free_space

            # Copy the group to new location
            unsafe_copyto!(m, next_free_space, m, group_posm2+2, 2group_length-2)

            # Adjust the pos entries in edit_map (bad memory order TODO: consider unzipping edit map to improve locality here)
            delta = next_free_space-(group_posm2+2)
            for k in 1:group_length-1
                target = m[next_free_space+2k-1]
                l = 2target + 10490
                m[l] += delta
            end

            # Mark the old group as moved so compaction will skip over it TODO: test this
            # TODO for perf: delete this and instead have compaction check if the index
            # pointed to by the start of the group points back (in the edit map) to that location
            if allocated_size != 0
                m[group_posm2 + 3] = unsigned(Int64(-allocated_size))
            end

            # update group start location
            group_posm2 = m[group_length_index-1] = next_free_space-2
        end
    end

    # insert the element into the group
    group_lastpos = group_posm2+2group_length
    m[group_lastpos] = shifted_significand
    m[group_lastpos+1] = i

    # log the insertion location in the edit map
    m[j] = group_lastpos

    nothing
end

merge_uint64(x::UInt64, y::UInt64) = UInt128(x) | (UInt128(y) << 64)
split_uint128(x::UInt128) = (x % UInt64, (x >>> 64) % UInt64)
get_shifted_significand_sum_index(exponent::UInt64) = 5 + 3*2046 - exponent >> 51
get_UInt128(m::Memory, i::Integer) = get_UInt128(m, _convert(Int, i))
get_UInt128(m::Memory, i::Int) = merge_uint64(m[i], m[i+1])
set_UInt128!(m::Memory, v::UInt128, i::Integer) = m[i:i+1] .= split_uint128(v)
"computes shifted_significand_sum<<(exponent_bits+shift) rounded up"
function compute_weight(m::Memory, exponent::UInt64, shifted_significand_sum::UInt128)
    shift = signed(exponent >> 52 + m[3])
    if shifted_significand_sum != 0 && Base.top_set_bit(shifted_significand_sum)+shift > 64
        # if this would overflow, drop shift so that it renormalizes down to 48.
        # this drops shift at least ~16 and makes the sum of weights at least ~2^48.
        # TODO for perf, don't check this on re-compute
        # Base.top_set_bit(shifted_significand_sum)+shift == 48
        # Base.top_set_bit(shifted_significand_sum)+signed(exponent >> 52 + m[3]) == 48
        # Base.top_set_bit(shifted_significand_sum)+signed(exponent >> 52) + signed(m[3]) == 48
        # signed(m[3]) == 48 - Base.top_set_bit(shifted_significand_sum) - signed(exponent >> 52)
        m3 = 48 - Base.top_set_bit(shifted_significand_sum) - exponent >> 52
        set_global_shift_decrease!(m, m3) # TODO for perf: special case all callsites to this function to take advantage of known shift direction and/or magnitude; also try outlining
        shift = signed(exponent >> 52 + m3)
    end
    weight = UInt64(shifted_significand_sum<<shift) # TODO for perf: change to % UInt64
    # round up
    weight += (trailing_zeros(shifted_significand_sum)+shift < 0) & (shifted_significand_sum != 0) # TODO for perf: ensure this final clause is const-prop eliminated when it can be (i.e. any time other than setting a weight to zero)
    weight
end
function update_weights!(m::Memory, exponent::UInt64, shifted_significand_sum::UInt128)
    weight = compute_weight(m, exponent, shifted_significand_sum)
    weight_index = 5 + 0x7fe - exponent >> 52
    old_weight = m[weight_index]
    m[weight_index] = weight
    m4 = m[4]
    m4 -= old_weight
    m4, o = Base.add_with_overflow(m4, weight)
    if o
        # If weights overflow (>2^64) then shift down by 16 bits
        m3 = m[3]-0x10
        set_global_shift_decrease!(m, m3, m4) # TODO for perf: special case all callsites to this function to take advantage of known shift direction and/or magnitude; also try outlining
        if weight_index < m[2] # if the new weight was not adjusted by set_global_shift_decrease!, then adjust it manually
            shift = signed(2051-weight_index+m3)
            new_weight = (shifted_significand_sum<<shift) % UInt64
            # round up
            new_weight += trailing_zeros(shifted_significand_sum)+shift < 0

            @check shifted_significand_sum != 0
            @check m[weight_index] == weight

            m[weight_index] = new_weight
            m[4] += new_weight-weight
        end
    else
        m[4] = m4
    end
end

function set_global_shift_increase!(m::Memory, m3::UInt64, m4, j0) # Increase shift, on deletion of elements
    @check signed(m[3]) < signed(m3)
    m[3] = m3
    # Story:
    # In the likely case that the weight decrease resulted in a level's weight hitting zero
    # that level's weight is already updated and m[4] adjusted accordingly TODO for perf don't adjust, pass the values around instead
    # In any event, m4 is accurate for current weights and all weights and sss's above (before) i0 are zero so we don't need to touch them
    # Between i0 and i1, weights that were previously 1 may need to be increased. Below (past, after) i1, all weights will round up to 1 or 0 so we don't need to touch them
    i0 = (j0 - 2041) >> 1

    # i1 is the lowest number such that for all i > i1, typemax(UInt128) (and therefore anything lower) will result in a weight of 1 (or 0 in the case of sss=0).
    #= TODO for clarity: delete this overlong comment
    weight = (typemax(UInt128)<<shift) % UInt64
    weight += (trailing_zeros(typemax(UInt128))+shift < 0) & (typemax(UInt128) != 0)
    weight == 1

    weight = (typemax(UInt128)<<shift) % UInt64
    weight += shift < 0
    weight == 1

    # shift should be < 0

    weight = (typemax(UInt128)<<shift) % UInt64
    weight += 1
    weight == 1

    (typemax(UInt128)<<shift) % UInt64 == 0

    (typemax(UInt128)>>-shift) % UInt64 == 0

    -shift >= 128

    -128 >= shift

    shift <= -128

    shift = signed(2051-i+m3)
    shift <= -128


    signed(2051-i+m3) <= -128
    signed(2051)-signed(i)+signed(m3) <= -128
    signed(2051)+signed(m3)+128 <= signed(i)
    signed(2051+128)+signed(m3) <= signed(i)

    2051+128+signed(m3) <= i
    =#
    # So for all i >= 2051+128+signed(m3), this holds. This means i1 = 2051+128+signed(m3)-1.
    i1 = min(2051+128+signed(m3)-1, 2050)

    for i in i0:i1 # TODO using i1-1 here passes tests (and is actually valid, I think. using i1-2 may fail if there are about 2^63 elements in the (i1-1)^th level. It would be possible to scale this range with length (m[1]) in which case testing could be stricter and performance could be (marginally) better, though not in large cases so possibly not worth doing at all)
        j = 2i+2041
        shifted_significand_sum = get_UInt128(m, j)
        shifted_significand_sum == 0 && continue # in this case, the weight was and still is zero
        shift = signed(2051-i+m3)
        weight = (shifted_significand_sum<<shift) % UInt64
        # round up
        weight += (trailing_zeros(shifted_significand_sum)+shift < 0)

        old_weight = m[i]
        m[i] = weight
        m4 += weight-old_weight
    end

    m[4] = m4
end

function set_global_shift_decrease!(m::Memory, m3::UInt64, m4=m[4]) # Decrease shift, on insertion of elements
    m3_old = m[3]
    m[3] = m3
    @check signed(m3) < signed(m3_old)

    # In the case of adding a giant element, call this first, then add the element.
    # In any case, this only adjusts elements at or after m[2]
    # from m[2] to the last index that could have a weight > 1 (possibly empty), recompute weights.
    # from max(m[2], the first index that can't have a weight > 1) to the last index that previously could have had a weight > 1, (never empty), set weights to 1 or 0
    m2 = signed(m[2])
    i1 = 2051+128+signed(m3)-1 # see above, this is the last index that could have weight > 1 (anything after this will have weight 1 or 0)
    i1_old = 2051+128+signed(m3_old)-1 # anything after this is already weight 1 or 0
    recompute_range = m2:min(i1, 2050)
    flatten_range = max(m2, i1+1):min(i1_old, 2050)
    @check length(recompute_range) <= 128 # TODO for perf: why is this not 64?
    @check length(flatten_range) <= 128 # TODO for perf: why is this not 64?

    for i in recompute_range
        j = 2i+2041
        shifted_significand_sum = get_UInt128(m, j)
        shift = signed(2051-i+m3)
        weight = (shifted_significand_sum<<shift) % UInt64
        # round up
        weight += (trailing_zeros(shifted_significand_sum)+shift < 0) & (shifted_significand_sum != 0) # TODO for perf: ensure this final clause is const-prop eliminated when it can be (i.e. any time other than setting a weight to zero)

        old_weight = m[i]
        m[i] = weight
        m4 += weight-old_weight
    end
    for i in flatten_range # set nonzeros to 1
        old_weight = m[i]
        weight = old_weight != 0
        m[i] = weight
        m4 += weight-old_weight
    end

    m[4] = m4
end

get_alloced_indices(exponent::UInt64) = 10491 - exponent >> 55, exponent >> 49 & 0x38

function _set_to_zero!(m::Memory, i::Int)
    # Find the entry's pos in the edit map table
    j = 2i + 10490
    pos = m[j]
    pos == 0 && return # if the entry is already zero, return
    # set the entry to zero (no need to zero the exponent)
    # m[j] = 0 is moved to after we adjust the edit_map entry for the shifted element, in case there is no shifted element
    exponent = m[j+1]

    # update group total weight and total weight
    shifted_significand_sum_index = get_shifted_significand_sum_index(exponent)
    shifted_significand_sum = get_UInt128(m, shifted_significand_sum_index)
    shifted_significand = m[pos]
    shifted_significand_sum -= shifted_significand
    set_UInt128!(m, shifted_significand_sum, shifted_significand_sum_index)

    weight_index = 5 + 0x7fe - exponent >> 52
    old_weight = m[weight_index]
    m4 = m[4]
    m4 -= old_weight
    if shifted_significand_sum == 0 # We zeroed out a group
        m[weight_index] = 0
        if m4 == 0 # There are no groups left
            m[2] = 2051
        else
            m2 = Int(m[2])
            if weight_index == m2 # We zeroed out the first group
                m[10235] != 0 && firstindex(m) <= m2 < 10235 && m2 isa Int || error() # This makes the following @inbounds safe. If the comiler can follow my reasoning, then the error checking can also improive effect analysis and therefore performance.
                while true # Update m[2]
                    m2 += 1
                    @inbounds m[m2] != 0 && break # TODO, see if the compiler can infer noub
                end
                m[2] = m2
            end
        end
    else # We did not zero out a group
        shift = signed(exponent >> 52 + m[3])
        new_weight = UInt64(shifted_significand_sum<<shift) # TODO for perf: change to % UInt64
        # round up
        new_weight += trailing_zeros(shifted_significand_sum)+shift < 0
        m[weight_index] = new_weight
        m4 += new_weight
    end

    if 0 < m4 < UInt64(1)<<32
        # If weights become less than 2^32 (but only if there are any nonzero weights), then for performance reasons (to keep the low probability rejection step sufficiently low probability)
        # Increase the shift to a reasonable level.
        # All nonzero true weights correspond to nonzero weights so 0 < m4 is a sufficient check to determine if we have fully emptied out the weights or not

        # TODO for perf: we can almost get away with loading only the most significant word of shifted_significand_sums. Here, we use the most significant 65 bits.
        j2 = 2m[2]+2041
        x = get_UInt128(m, j2)
        # TODO refactor indexing for simplicity
        x2 = UInt64(x>>63) #TODO for perf %UInt64
        @check x2 != 0
        for i in 1:Sys.WORD_SIZE # TODO for perf, we can get away with shaving 1 to 10 off of this loop.
            x2 += _convert(UInt, get_UInt128(m, j2+2i) >> (63+i))
        end

        # x2 is computed by rounding down at a certian level and then summing
        # m[4] will be computed by rounding up at a more precise level and then summing
        # x2 could be 1, composed of 1.9 + .9 + .9 + ... for up to about log2(length) levels
        # meaning m[4] could be up to 1+log2(length) times greater than predicted according to x2
        # if length is 2^64 than this could push m[4]'s top set bbit up to 8 bits higher.

        # If, on the other hand, x2 was computed with significantly higher precision, then
        # it could overflow if there were 2^64 elements in a weight. TODO: We could probably
        # squeeze a few more bits out of this, but targeting 46 with a window of 46 to 52 is
        # plenty good enough.

        m3 = -17 - Base.top_set_bit(x2) - (6143-j2)>>1
        # TODO test that this actually achieves the desired shift and results in a new sum of about 2^48

        set_global_shift_increase!(m, m3, m4, j2) # TODO for perf: special case all callsites to this function to take advantage of known shift direction and/or magnitude; also try outlining

        @check 46 <= Base.top_set_bit(m[4]) <= 53 # Could be a higher because of the rounding up, but this should never bump top set bit by more than about 8 # TODO for perf: delete
    else
        m[4] = m4
    end

    # lookup the group by exponent
    group_length_index = shifted_significand_sum_index + 2*2046 + 1
    group_posm2 = m[group_length_index-1]
    group_length = m[group_length_index]
    group_lastpos = group_posm2+2group_length

    # TODO for perf: see if it's helpful to gate this on pos != group_lastpos
    # shift the last element of the group into the spot occupied by the removed element
    m[pos] = m[group_lastpos]
    shifted_element = m[pos+1] = m[group_lastpos+1]

    # adjust the edit map entry of the shifted element
    m[2shifted_element + 10490] = pos
    m[j] = 0

    # When zeroing out a group, mark the group as empty so that compaction will update the group metadata and then skip over it.
    if shifted_significand_sum == 0
        m[group_posm2+3] = exponent>>52 | 0x8000000000000000
    end

    # shrink the group
    m[group_length_index] = group_length-1 # no need to zero group entries

    nothing
end


ResizableWeights(len::Integer) = ResizableWeights(FixedSizeWeights(len))
SemiResizableWeights(len::Integer) = SemiResizableWeights(FixedSizeWeights(len))
function FixedSizeWeights(len::Integer)
    m = Memory{UInt64}(undef, allocated_memory(len))
    m .= 0 # TODO for perf: delete this. It's here so that a sparse rendering for debugging is easier TODO for tests: set this to 0xdeadbeefdeadbeed
    m[4:10491+2len] .= 0 # metadata and edit map need to be zeroed but the bulk does not
    m[1] = len
    m[2] = 2051
    # no need to set m[3]
    m[10235] = 10492+2len
    _FixedSizeWeights(m)
end
allocated_memory(length::Integer) = 10491 + 8*length # TODO for perf: consider giving some extra constant factor allocation to avoid repeated compaction at small sizes
length_from_memory(allocated_memory::Integer) = (allocated_memory-10491) >> 3 # Int((allocated_memory-10491)/8)

Base.resize!(w::Union{SemiResizableWeights, ResizableWeights}, len::Integer) = resize!(w, Int(len))
function Base.resize!(w::Union{SemiResizableWeights, ResizableWeights}, len::Int)
    m = w.m
    old_len = m[1]
    if len > old_len
        am = allocated_memory(len)
        if am > length(m)
            w isa SemiResizableWeights && throw(ArgumentError("Cannot increase the size of a SemiResizableWeights above its original allocated size. Try using a ResizableWeights instead."))
            _resize!(w, len)
        else
            m[1] = len
        end
    else
        w[len+1:old_len] .= 0 # This is a necessary but highly nontrivial operation
        m[1] = len
    end
    w
end
"""
Reallocate w with the size len, compacting w into that new memory.
Any elements if w past len must be set to zero already (that's a general invariant for
Weigths, though, not just this function).
"""
function _resize!(w::ResizableWeights, len::Integer)
    m = w.m
    old_len = m[1]
    m2 = Memory{UInt64}(undef, allocated_memory(len))
    m2 .= 0 # For debugging; TODO: delete, TODO: set to 0xdeadbeefdeadbeef to test
    m2[1] = len
    if len > old_len # grow
        unsafe_copyto!(m2, 2, m, 2, 2old_len + 10490)
        m2[2old_len + 10492:2len + 10491] .= 0
    else # shrink
        unsafe_copyto!(m2, 2, m, 2, 2len + 10490)
    end

    compact!(m2, Int(2len + 10492), m, Int(2length_from_memory(length(m)) + 10492))
    w.m = m2
    w
end

function compact!(dst::Memory{UInt64}, dst_i::Int, src::Memory{UInt64}, src_i::Int)
    next_free_space = src[10235]

    while src_i < next_free_space

        # Skip over abandoned groups TODO refactor these loops for clarity
        target = signed(src[src_i+1])
        while target < 0
            if unsigned(target) < 0xc000000000000000 # empty non-abandoned group; let's clean it up
                @check 0x8000000000000001 <= unsigned(target) <= 0x80000000000007fe
                exponent = unsigned(target) << 52 # TODO for clarity: dry this
                allocs_index,allocs_subindex = get_alloced_indices(exponent)
                allocs_chunk = dst[allocs_index] # TODO for perf: consider not copying metadata on out of place compaction (and consider the impact here)
                log2_allocated_size_p1 = allocs_chunk >> allocs_subindex % UInt8
                allocated_size = 1<<(log2_allocated_size_p1-1)
                new_chunk = allocs_chunk - UInt64(log2_allocated_size_p1) << allocs_subindex
                dst[allocs_index] = new_chunk # zero out allocated size (this will force re-allocation so we can let the old, wrong pos info stand)
                src_i += 2allocated_size # skip the group
            else # the decaying corpse of an abandoned group. Ignore it.
                src_i -= 2target
            end
            src_i >= next_free_space && @goto break_outer
            target = signed(src[src_i+1])
        end

        # Trace an element of the group back to the edit info table to find the group id
        j = 2target + 10490
        exponent = src[j+1]

        # Lookup the group in the group location table to find its length (performance optimization for copying, necesary to decide new allocated size and update pos)
        # exponent of 0x7fe0000000000000 is index 6+3*2046
        # exponent of 0x0010000000000000 is index 4+5*2046
        group_length_index = 6 + 5*2046 - exponent >> 51
        group_length = src[group_length_index]

        # Update group pos in level_location_info
        dst[group_length_index-1] += unsigned(Int64(dst_i-src_i))

        # Lookup the allocated size (an alternative to scanning for the next nonzero, needed because we are setting allocated size)
        # exponent of 0x7fe0000000000000 is index 6+5*2046, 2
        # exponent of 0x7fd0000000000000 is index 6+5*2046, 1
        # exponent of 0x0040000000000000 is index 5+5*2046+512, 0
        # exponent of 0x0030000000000000 is index 5+5*2046+512, 3
        # exponent of 0x0020000000000000 is index 5+5*2046+512, 2
        # exponent of 0x0010000000000000 is index 5+5*2046+512, 1
        # allocs_index = 5+5*2046+512 - (exponent >> 54), (exponent >> 52) & 0x3
        # allocated_size = 2 << ((src[allocs_index[1]] >> (8allocs_index[1])) % UInt8)
        allocs_index,allocs_subindex = get_alloced_indices(exponent)
        allocs_chunk = dst[allocs_index]
        log2_allocated_size = allocs_chunk >> allocs_subindex % UInt8 - 1
        log2_new_allocated_size = group_length == 0 ? 0 : Base.top_set_bit(group_length-1)
        new_chunk = allocs_chunk + Int64(log2_new_allocated_size - log2_allocated_size) << allocs_subindex
        dst[allocs_index] = new_chunk

        # Copy the group to a compacted location
        unsafe_copyto!(dst, dst_i, src, src_i, 2group_length)

        # Adjust the pos entries in edit_map (bad memory order TODO: consider unzipping edit map to improve locality here)
        delta = unsigned(Int64(dst_i-src_i))
        dst[j] += delta
        for k in 1:signed(group_length)-1
            target = src[src_i+2k+1]
            j = 2target + 10490
            dst[j] += delta
        end

        # Advance indices
        src_i += 2*1<<log2_allocated_size # TODO add test that fails if the 2* part is removed
        dst_i += 2*1<<log2_new_allocated_size
    end
    @label break_outer
    dst[10235] = dst_i
end

# Conform to the AbstractArray API
Base.size(w::Weights) = (w.m[1],)

# Shoe-horn into the legacy DynamicDiscreteSampler API so that we can leverage existing tests
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
Base.rand(rng::AbstractRNG, wbs::WeightBasedSampler) = rand(rng, wbs.w)
Base.rand(rng::AbstractRNG, wbs::WeightBasedSampler, n::Integer) = [rand(rng, wbs.w) for _ in 1:n]

const DynamicDiscreteSampler = WeightBasedSampler

end
