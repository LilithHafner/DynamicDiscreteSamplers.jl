module DynamicDiscreteSamplers

export DynamicDiscreteSampler

using Random

isdefined(@__MODULE__, :Memory) || const Memory = Vector # Compat for Julia < 1.11

const DEBUG = Base.JLOptions().check_bounds == 1
_convert(T, x) = DEBUG ? T(x) : x%T

"""
    Weights <: AbstractVector{Float64}

An abstract vector capable of storing normal, non-negative floating point numbers on which
`rand` samples an index according to values rather than sampling a value uniformly.
"""
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

#===== Overview  ======

# Objective

This package provides a discrete random sampler with the following key properties
 - Exact: sampling probability exactly matches provided weights
 - O(1) worst case expected runtime for sampling
        (though termination only guaranteed probabilistically)
 - O(1) worst case amortized update time to change the weight of any element
        (an individual update may take up to O(n))
 - O(n) space complexity
 - O(n) construction time
 - Fast constant factor in practice. Typical usage has a constant factor of tens of clock
        cycles and pathological usage has a constant factor of thousands of clock cycles.


# Brief implementation overview

Weights are are divided into levels according to their exponents. To sample, first sample a
level and then sample an element within that level.


# Definition of terms

v::Float64 aka weight
    An entry in a Weights object set with `w[i] = v`, retrieved with `v = w[i]`.
exponent::UInt64
    The exponent of a weight is `reinterpret(UInt64, weight) >> 52`.
    Note that this is _not_ the same as `Base.exponent(weight)` nor
    `reinterpret(UInt64, weight) & Base.exponent_mask(Float64)`.
level
    All the weights in a Weights object that have the same exponent.
significand::UInt64
    The significand of a weight is `reinterpret(UInt64, weight) << 11 | 0x8000000000000000`.
    Ranges from 0x8000000000000000 for 1.0 to 0xfffffffffffff800 for 1.9999...
    The ratio of two weights with the same exponent is the ratio of their significands.
significand_sum::UInt128
    The sum of the significands of the weights in a given level.
    Is 0 for empty levels and otherwise ranges from widen(0x8000000000000000) to
    widen(0xfffffffffffff800) * length(weights).
weight
    Refers to the relative likely hood of an event. Weights are like probabilities except
    they do not need to sum to one. In this codebase, the term "weight" is used to refer to
    four things: the weight of an element relative to all the other elements in in a
    Weights object; the weight of an element relative to the other elements in its level;
    the weight of a level relative to the other levels as defined by level_weights; and the
    weight of a level relative to the other levels as defined by significand_sums.


# Implementation and data structure overview

Weights are normal, non-negative Float64s. They are divided into levels according to their
exponents. Each level has a weight which is the exact sum of the weights in that level. We
can't represent this sum exactly as a Float64 so we represent it as significand_sum::UInt128
which is the sum of the significands of the weights in that level. To get the level's weight,
compute big(significand_sum)<<exponent.

## Sampling
Sampling with BigInt weights is not efficient so each level also has an approximate weight
which is a UInt64. These approximate weights are computed as exact_weight<<global_shift+1 if
exact_weight is nonzero, and 0 otherwise. global_shift is a constant maintained by the
sampler so that the sum of the approximate weights is less than 2^64 and greater than 2^32.

To sample a level, we pick a random UInt64 between 1 and the sum of the approximate weights.
Then use linear search to find the level that corresponds to (with the highest weight levels
at the start of that search). This picks a level with probability according to approximate
weights which is not quite accurate. We correct for this by adding a small probability
rejection. If the linear search lands on the edge of a level (which can happen at most
2046/2^32 of the time), we consider rejecting. That process need not be fast but is O(1) and
utilizes significand_sum directly.

Sampling an element within a level is straightforward rejection sampling which is O(1)
because all rejection probabilities are less than or equal to 1/2.

## Stored values and invariants
TODO

## Updates
TODO


# Memory layout (TODO: add alternative layout for small cases) =#

# <memory_length::Int>
# 1                      length::Int
# 2                      max_level::Int # absolute pointer to the first element of level weights that is nonzero
# 3                      shift::Int level weights are equal to significand_sums<<(exponent+shift), plus one if significand_sum is not zero
# 4                      sum(level weights)::UInt64
# 5..2050                level weights::[UInt64 2046] # earlier is higher. first is exponent 0x7fe, last is exponent 0x001. Subnormal are not supported.
# 2051..6142             significand_sums::[UInt128 2046] # sum of significands (the maximum significand contributes 0xfffffffffffff800)
# 6143..10234            level location info::[NamedTuple{pos::Int, length::Int} 2046] indexes into sub_weights, pos is absolute into m.

# gc info:
# 10235                  next_free_space::Int (used to re-allocate) <index 10235>
# 16 unused bits
# 10236..10491           level allocated length::[UInt8 2046] (2^(x-1) is implied)

# 10492..10491+2len      edit_map (maps index to current location in sub_weights)::[pos::Int, exponent::UInt64] (zero means zero; fixed location, always at the start. Force full realloc when it OOMs. TODO for perf: exponent could be UInt11, lots of wasted bits)

# 10492+2allocated_len..10491+2allocated_len+6len sub_weights (woven with targets)::[[significand::UInt64, target::Int}]]. allocated_len == length_from_memory(length(m))

# significands are stored in sub_weights with their implicit leading 1 added
#     element_from_sub_weights = 0x8000000000000000 | (reinterpret(UInt64, weight::Float64) << 11)
# And sampled with
#     rand(UInt64) < element_from_sub_weights
# this means that for the lowest normal significand (52 zeros with an implicit leading one),
# achieved by 2.0, 4.0, etc the significand stored in sub_weights is 0x8000000000000000
# and there are 2^63 pips less than that value (1/2 probability). For the
# highest normal significand (52 ones with an implicit leading 1) the significand
# stored in sub_weights is 0xfffffffffffff800 and there are 2^64-2^11 pips less than
# that value for a probability of (2^64-2^11) / 2^64 == (2^53-1) / 2^53 == prevfloat(2.0)/2.0
# significand sums are literal sums of the element_from_sub_weights's (though stored
# as UInt128s because any two element_from_sub_weights's will overflow when added).

# target can also store metadata useful for compaction.
# the range 0x0000000000000001 to 0x7fffffffffffffff (1:typemax(Int)) represents literal targets
# the range 0x8000000000000001 to 0x80000000000007fe indicates that this is an empty but non-abandoned group with exponent target-0x8000000000000000
# the range 0xc000000000000000 to 0xffffffffffffffff indicates that the group is abandoned and has length -target.

## Initial API:

# setindex!, getindex, resize! (auto-zeros), scalar rand
# Trivial extensions:
# push!, delete!

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
    if x == mi # mi is the weight rounded down plus 1. If they are equal than we should refine further and possibly reject. This branch is very uncommon and still O(1); constant factors don't matter here.
        # shift::Int = exponent+m[3]
        # significand_sum::UInt128 = ...
        # weight::UInt64 = significand_sum<<shift+1
        # true_weight::ExactReal = exact(significand_sum)<<shift
        # true_weight::ExactReal = significand_sum<<shift + exact(significand_sum)<<shift & ...0000.1111...
        # rejection_p = weight-true_weight = (significand_sum<<shift+1) - (significand_sum<<shift + exact(significand_sum)<<shift & ...0000.1111...)
        # rejection_p = 1 - exact(significand_sum)<<shift & ...0000.1111...
        # acceptance_p = exact(significand_sum)<<shift & ...0000.1111...  (for example, if significand_sum<<shift is exact, then acceptance_p will be zero)
        # TODO for confidence: add a test that fails if this were to mix up floor+1 and ceil.
        j = 2i+2041
        exponent = 0x7fe+5-i
        shift = signed(exponent + m[3])
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
    pos = m[j]
    len = m[j+1]

    # Sample within level
    while true
        r = rand(rng, UInt64)
        k1 = (r>>leading_zeros(len-1))
        k2 = k1<<1+pos # TODO for perf: try %Int here (and everywhere)
        # TODO for perf: delete the k1 < len check by maintaining all the out of bounds m[k2] equal to 0
        k1 < len && rand(rng, UInt64) < m[k2] && return Int(signed(m[k2+1]))
    end
end

function _getindex(m::Memory{UInt64}, i::Int)
    @boundscheck 1 <= i <= m[1] || error()#throw(BoundsError(_FixedSizeWeights(m), i))
    j = 2i + 10490
    pos = m[j]
    pos == 0 && return 0.0
    exponent = m[j+1]
    weight = m[pos]
    reinterpret(Float64, (exponent<<52) | (weight - 0x8000000000000000) >> 11)
end

function _setindex!(m::Memory, v::Float64, i::Int)
    @boundscheck 1 <= i <= m[1] || error()#throw(BoundsError(_FixedSizeWeights(m), i))
    uv = reinterpret(UInt64, v)
    if uv == 0
        _set_to_zero!(m, i)
        return
    end
    0x0010000000000000 <= uv <= 0x7fefffffffffffff || error()#throw(DomainError(v, "Invalid weight")) # Excludes subnormals

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
    # TODO for performance: join these two operations
    _set_to_zero!(m, i)
    _set_from_zero!(m, v, i)
end

function _set_from_zero!(m::Memory, v::Float64, i::Int)
    uv = reinterpret(UInt64, v)
    j = 2i + 10490

    exponent = uv >> 52
    m[j+1] = exponent
    # update group total weight and total weight
    significand_sum_index = get_significand_sum_index(exponent)
    significand_sum = get_UInt128(m, significand_sum_index)
    significand = 0x8000000000000000 | uv << 11
    significand_sum += significand
    set_UInt128!(m, significand_sum, significand_sum_index)
    weight_index = 5 + 0x7fe - exponent
    if m[4] == 0 # if we were empty, set global shift (m[3]) so that m[4] will become ~2^40.
        m[3] = -24 - exponent

        shift = -24
        weight = UInt64(significand_sum<<shift) + 1 # TODO for perf: change to % UInt64

        m[weight_index] = weight
        m[4] = weight
    else
        shift = signed(exponent + m[3])
        if Base.top_set_bit(significand_sum)+shift > 64
            # if this would overflow, drop shift so that it renormalizes down to 48.
            # this drops shift at least ~16 and makes the sum of weights at least ~2^48. # TODO: add an assert
            # Base.top_set_bit(significand_sum)+shift == 48
            # Base.top_set_bit(significand_sum)+signed(exponent + m[3]) == 48
            # Base.top_set_bit(significand_sum)+signed(exponent) + signed(m[3]) == 48
            # signed(m[3]) == 48 - Base.top_set_bit(significand_sum) - signed(exponent)
            m3 = 48 - Base.top_set_bit(significand_sum) - exponent
            set_global_shift_decrease!(m, m3) # TODO for perf: special case all call sites to this function to take advantage of known shift direction and/or magnitude; also try outlining
            shift = signed(exponent + m3)
        end
        weight = UInt64(significand_sum<<shift) + 1 # TODO for perf: change to % UInt64

        weight_index = 5 + 0x7fe - exponent
        old_weight = m[weight_index]
        m[weight_index] = weight
        m4 = m[4]
        m4 -= old_weight
        m4, o = Base.add_with_overflow(m4, weight)
        if o
            # If weights overflow (>2^64) then shift down by 16 bits
            m3 = m[3]-0x10
            set_global_shift_decrease!(m, m3, m4) # TODO for perf: special case all call sites to this function to take advantage of known shift direction and/or magnitude; also try outlining
            if weight_index < m[2] # if the new weight was not adjusted by set_global_shift_decrease!, then adjust it manually
                shift = signed(2051-weight_index+m3)
                new_weight = (significand_sum<<shift) % UInt64 + 1
                m[weight_index] = new_weight
                m[4] += new_weight-weight
            end
        else
            m[4] = m4
        end
    end
    m[2] = min(m[2], weight_index) # Set after insertion because update_weights! may need to update the global shift, in which case knowing the old m[2] will help it skip checking empty levels

    # lookup the group by exponent and bump length
    group_length_index = significand_sum_index + 2*2046 + 1
    group_pos = m[group_length_index-1]
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
        if next_free_space == (group_pos-2)+2group_length # note that this is valid even if group_length is 1 (previously zero).
            new_allocation_length = max(2, 2allocated_size)
            new_next_free_space = next_free_space+new_allocation_length
            if new_next_free_space > length(m)+1 # There isn't room; we need to compact
                m[group_length_index] = group_length-1 # See comment above; we don't want to copy past the end of m
                next_free_space = compact!(m, m)
                group_pos = next_free_space-new_allocation_length # The group will move but remian the last group
                new_next_free_space = next_free_space+new_allocation_length
                m[group_length_index] = group_length

                # Re-lookup allocated chunk because compaction could have changed other
                # chunk elements. However, the allocated size of this group could not have
                # changed because it was previously maxed out.
                allocs_chunk = m[allocs_index]
            end
            # expand the allocated size and bump next_free_space
            new_chunk = allocs_chunk + UInt64(1) << allocs_subindex
            m[allocs_index] = new_chunk
            m[10235] = new_next_free_space
        else # move and reallocate (this branch also handles creating new groups: TODO expirment with perf and clarity by splicing that branch out)
            twice_new_allocated_size = max(0x2,allocated_size<<2)
            new_next_free_space = next_free_space+twice_new_allocated_size
            if new_next_free_space > length(m)+1 # out of space; compact. TODO for perf, consider resizing at this time slightly eagerly?
                m[group_length_index] = group_length-1 # incrementing the group length before compaction is spotty because if the group was previously empty then this new group length will be ignored (compact! loops over sub_weights, not levels)
                next_free_space = compact!(m, m)
                m[group_length_index] = group_length
                new_next_free_space = next_free_space+twice_new_allocated_size

                group_pos = m[group_length_index-1] # The group likely moved during compaction

                # Re-lookup allocated chunk because compaction could have changed other
                # chunk elements. However, the allocated size of this group could not have
                # changed because it was previously maxed out.
                allocs_chunk = m[allocs_index]
            end
            # TODO for perf: make compact! re-allocate the expanded group larger so there's no need to double the allocated size here if the compact branch is taken
            # TODO for perf, try removing the moveie before compaction (tricky: where to store that info?)
            # TODO make this whole alg dry, but only after setting up robust benchmarks in CI

            # expand the allocated size and bump next_free_space
            new_chunk = allocs_chunk + UInt64(1) << allocs_subindex
            m[allocs_index] = new_chunk

            m[10235] = new_next_free_space

            # Copy the group to new location
            (v"1.11" <= VERSION || 2group_length-2 != 0) && copyto!(m, next_free_space, m, group_pos, 2group_length-2) # TODO for clarity and maybe perf: remove this version check

            # Adjust the pos entries in edit_map (bad memory order TODO: consider unzipping edit map to improve locality here)
            delta = next_free_space-group_pos
            for k in 1:group_length-1
                target = m[next_free_space+2k-1]
                l = 2target + 10490
                m[l] += delta
            end

            # Mark the old group as moved so compaction will skip over it TODO: test this
            # TODO for perf: delete this and instead have compaction check if the index
            # pointed to by the start of the group points back (in the edit map) to that location
            if allocated_size != 0
                m[group_pos+1] = unsigned(Int64(-allocated_size))
            end

            # update group start location
            group_pos = m[group_length_index-1] = next_free_space
        end
    end

    # insert the element into the group
    group_lastpos = (group_pos-2)+2group_length
    m[group_lastpos] = significand
    m[group_lastpos+1] = i

    # log the insertion location in the edit map
    m[j] = group_lastpos

    nothing
end

merge_uint64(x::UInt64, y::UInt64) = UInt128(x) | (UInt128(y) << 64)
split_uint128(x::UInt128) = (x % UInt64, (x >>> 64) % UInt64)
get_significand_sum_index(exponent::UInt64) = 5 + 3*2046 - 2exponent
get_UInt128(m::Memory, i::Integer) = get_UInt128(m, _convert(Int, i))
get_UInt128(m::Memory, i::Int) = merge_uint64(m[i], m[i+1])
function set_UInt128!(m::Memory, v::UInt128, i::Integer)
    x,y = split_uint128(v)
    m[i] = x
    m[i+1] = y
    nothing
end

function set_global_shift_increase!(m::Memory, m2, m3::UInt64, m4) # Increase shift, on deletion of elements
    m[3] = m3
    # Story:
    # In the likely case that the weight decrease resulted in a level's weight hitting zero
    # that level's weight is already updated and m4 adjusted accordingly TODO for perf don't adjust, pass the values around instead
    # In any event, m4 is accurate for current weights and all weights and significand_sums's above (before) m2 are zero so we don't need to touch them
    # Between m2 and i1, weights that were previously 1 may need to be increased. Below (past, after) i1, all weights will round up to 1 or 0 so we don't need to touch them

    #=
    weight = UInt64(significand_sum<<shift) + 1
    when is that always 1? when
    UInt64(significand_sum<<shift) == 0
    and because shift < 0 and significand_sum could be as much as 2^64*2^64/8/8-1 = 2^122-1,
    shift <= -122
    shift = signed(2051-i+m3)
    signed(2051-i+m3) <= -122
    2173+signed(m3) <= i
    So for i < 2173+signed(m3), we could need to adjust the ith weight
    =#
    recompute_range = signed(m2):min(2172+signed(m3), 2050) # TODO It would be possible to scale this range with length (m[1]) in which case testing could be stricter and performance could be (marginally) better, though not in large cases so possibly not worth doing at all)
    m[4] = recompute_weights!(m, m3, m4, recompute_range)
end

function set_global_shift_decrease!(m::Memory{UInt64}, m3::UInt64, m4::UInt64=m[4]) # Decrease shift, on insertion of elements
    m3_old = m[3]
    m[3] = m3

    # In the case of adding a giant element, call this first, then add the element.
    # In any case, this only adjusts elements at or after m[2]
    # from m[2] to the last index that could have a weight > 1 (possibly empty), recompute weights.
    # from max(m[2], the first index that can't have a weight > 1) to the last index that previously could have had a weight > 1, (never empty), set weights to 1 or 0
    m2 = signed(m[2])
    i1 = 2172+signed(m3) # see above, this is the last index that could have weight > 1 (anything after this will have weight 1 or 0)
    i1_old = 2172+signed(m3_old) # anything after this is already weight 1 or 0
    recompute_range = signed(m2):min(i1, 2050)
    flatten_range = max(m2, i1+1):min(i1_old, 2050)
    # From the level where one element contributes 2^64 to the level where one element contributes 1 is 64, and from there to the level where 2^64 elements contributes 1 is another 2^64.

    m4 = recompute_weights!(m, m3, m4, recompute_range)
    checkbounds(m, flatten_range)
    for i in flatten_range # set nonzeros to 1
        Base.@assume_effects :noub @inbounds old_weight = m[i]
        weight = old_weight != 0
        Base.@assume_effects :noub @inbounds m[i] = weight
        m4 += weight-old_weight
    end

    m[4] = m4
end

function recompute_weights!(m::Memory{UInt64}, m3::UInt64, m4::UInt64, range::UnitRange{Int})
    checkbounds(m, range)
    for i in range
        j = 2i+2041
        significand_sum = get_UInt128(m, j)
        significand_sum == 0 && continue # in this case, the weight was and still is zero
        shift = signed(2051-i+m3)
        weight = (significand_sum<<shift) % UInt64 + 1

        Base.@assume_effects :noub @inbounds old_weight = m[i]
        Base.@assume_effects :noub @inbounds m[i] = weight
        m4 += weight-old_weight
    end
    m4
end

get_alloced_indices(exponent::UInt64) = 10491 - exponent >> 3, exponent << 3 & 0x38

function _set_to_zero!(m::Memory, i::Int)
    # Find the entry's pos in the edit map table
    j = 2i + 10490
    pos = m[j]
    pos == 0 && return # if the entry is already zero, return
    # set the entry to zero (no need to zero the exponent)
    # m[j] = 0 is moved to after we adjust the edit_map entry for the shifted element, in case there is no shifted element
    exponent = m[j+1]

    # update group total weight and total weight
    significand_sum_index = get_significand_sum_index(exponent)
    significand_sum = get_UInt128(m, significand_sum_index)
    significand = m[pos]
    significand_sum -= significand
    set_UInt128!(m, significand_sum, significand_sum_index)

    weight_index = 5 + 0x7fe - exponent
    old_weight = m[weight_index]
    m4 = m[4]
    m4 -= old_weight
    if significand_sum == 0 # We zeroed out a group
        m[weight_index] = 0
        if m4 == 0 # There are no groups left
            m[2] = 2051
        else
            m2 = Int(m[2])
            if weight_index == m2 # We zeroed out the first group
                m[10235] != 0 && firstindex(m) <= m2 < 10235 && m2 isa Int && m isa Memory{UInt64} || error() # This makes the following @inbounds safe. If the compiler can follow my reasoning, then the error checking can also improve effect analysis and therefore performance.
                Base.@assume_effects :terminates_locally while true # Update m[2]
                    m2 += 1
                    Base.@assume_effects :noub @inbounds m[m2] != 0 && break # TODO help the compiler infer noub
                end
                m[2] = m2
            end
        end
    else # We did not zero out a group
        shift = signed(exponent + m[3])
        new_weight = UInt64(significand_sum<<shift) + 1# TODO for perf: change to % UInt64
        m[weight_index] = new_weight
        m4 += new_weight
    end

    if 0 < m4 < UInt64(1)<<32
        # If weights become less than 2^32 (but only if there are any nonzero weights), then for performance reasons (to keep the low probability rejection step sufficiently low probability)
        # Increase the shift to a reasonable level.
        # All nonzero significand_sums correspond to nonzero weights so 0 < m4 is a sufficient check to determine if we have fully emptied out the weights or not

        # TODO for perf: we can almost get away with loading only the most significant word of significand_sums. Here, we use the most significant 65 bits.
        m2 = m[2]
        j2 = 2m2+2041
        x = get_UInt128(m, j2)
        # TODO refactor indexing for simplicity
        x2 = UInt64(x>>63) #TODO for perf %UInt64
        Base.@assume_effects :terminates_locally for i in 1:Sys.WORD_SIZE # TODO for perf, we can get away with shaving 1 to 10 off of this loop.
            x2 += _convert(UInt, get_UInt128(m, j2+2i) >> (63+i))
        end

        # x2 is computed by rounding down at a certain level and then summing
        # m[4] will be computed by rounding up at a more precise level and then summing
        # x2 could be 1, composed of 1.9 + .9 + .9 + ... for up to about log2(length) levels
        # meaning m[4] could be up to 1+log2(length) times greater than predicted according to x2
        # if length is 2^64 than this could push m[4]'s top set bit up to 8 bits higher.

        # If, on the other hand, x2 was computed with significantly higher precision, then
        # it could overflow if there were 2^64 elements in a weight. TODO: We could probably
        # squeeze a few more bits out of this, but targeting 46 with a window of 46 to 52 is
        # plenty good enough.

        m3 = -17 - Base.top_set_bit(x2) - (6143-j2)>>1
        # TODO test that this actually achieves the desired shift and results in a new sum of about 2^48

        set_global_shift_increase!(m, m2, m3, m4) # TODO for perf: special case all call sites to this function to take advantage of known shift direction and/or magnitude; also try outlining

    else
        m[4] = m4
    end

    # lookup the group by exponent
    group_length_index = significand_sum_index + 2*2046 + 1
    group_pos = m[group_length_index-1]
    group_length = m[group_length_index]
    group_lastpos = (group_pos-2)+2group_length

    # TODO for perf: see if it's helpful to gate this on pos != group_lastpos
    # shift the last element of the group into the spot occupied by the removed element
    m[pos] = m[group_lastpos]
    shifted_element = m[pos+1] = m[group_lastpos+1]

    # adjust the edit map entry of the shifted element
    m[2shifted_element + 10490] = pos
    m[j] = 0

    # When zeroing out a group, mark the group as empty so that compaction will update the group metadata and then skip over it.
    if significand_sum == 0
        m[group_pos+1] = exponent | 0x8000000000000000
    end

    # shrink the group
    m[group_length_index] = group_length-1 # no need to zero group entries

    nothing
end


ResizableWeights(len::Integer) = ResizableWeights(FixedSizeWeights(len))
SemiResizableWeights(len::Integer) = SemiResizableWeights(FixedSizeWeights(len))
function FixedSizeWeights(len::Integer)
    m = Memory{UInt64}(undef, allocated_memory(len))
    # m .= 0 # TODO for perf: delete this. It's here so that a sparse rendering for debugging is easier TODO for tests: set this to 0xdeadbeefdeadbeed
    Base.@assume_effects :terminates_locally for i in 4:10491+2Int(len)
        m[i] = 0
    end
    # m[4:10491+2len] .= 0 # metadata and edit map need to be zeroed but the bulk does not
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
            w isa SemiResizableWeights && error()#throw(ArgumentError("Cannot increase the size of a SemiResizableWeights above its original allocated size. Try using a ResizableWeights instead."))
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
Weights, though, not just this function).
"""
function _resize!(w::ResizableWeights, len::Integer)
    m = w.m
    old_len = m[1]
    m2 = Memory{UInt64}(undef, allocated_memory(len))
    # m2 .= 0 # For debugging; TODO: delete, TODO: set to 0xdeadbeefdeadbeef to test
    m2[1] = len
    if len > old_len # grow
        copyto!(m2, 2, m, 2, 2old_len + 10490)
        for i in 2old_len + 10492:2len + 10491
            m2[i] = 0
        end
    else # shrink
        copyto!(m2, 2, m, 2, 2len + 10490)
    end

    compact!(m2, m)
    w.m = m2
    w
end

function compact!(dst::Memory{UInt64}, src::Memory{UInt64})
    dst_i = Int(2length_from_memory(length(dst)) + 10492)
    src_i = Int(2length_from_memory(length(src)) + 10492)
    next_free_space = src[10235]

    while src_i < next_free_space

        # Skip over abandoned groups TODO refactor these loops for clarity
        target = signed(src[src_i+1])
        while target < 0
            if unsigned(target) < 0xc000000000000000 # empty non-abandoned group; let's clean it up
                exponent = unsigned(target) - 0x8000000000000000 # TODO for clarity: dry this
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

        # Lookup the group in the group location table to find its length (performance optimization for copying, necessary to decide new allocated size and update pos)
        # exponent of 0x00000000000007fe is index 6+3*2046
        # exponent of 0x0000000000000001 is index 4+5*2046
        group_length_index = 6 + 5*2046 - 2exponent
        group_length = src[group_length_index]

        # Update group pos in level_location_info
        dst[group_length_index-1] += unsigned(Int64(dst_i-src_i))

        # Lookup the allocated size (an alternative to scanning for the next nonzero, needed because we are setting allocated size)
        # exponent of 0x00000000000007fe is index 6+5*2046, 2
        # exponent of 0x00000000000007fd is index 6+5*2046, 1
        # exponent of 0x0000000000000004 is index 5+5*2046+512, 0
        # exponent of 0x0000000000000003 is index 5+5*2046+512, 3
        # exponent of 0x0000000000000002 is index 5+5*2046+512, 2
        # exponent of 0x0000000000000001 is index 5+5*2046+512, 1
        allocs_index,allocs_subindex = get_alloced_indices(exponent)
        allocs_chunk = dst[allocs_index]
        log2_allocated_size = allocs_chunk >> allocs_subindex % UInt8 - 1
        log2_new_allocated_size = group_length == 0 ? 0 : Base.top_set_bit(group_length-1)
        new_chunk = allocs_chunk + Int64(log2_new_allocated_size - log2_allocated_size) << allocs_subindex
        dst[allocs_index] = new_chunk

        # Copy the group to a compacted location
        copyto!(dst, dst_i, src, src_i, 2group_length)

        # Adjust the pos entries in edit_map (bad memory order TODO: consider unzipping edit map to improve locality here)
        delta = unsigned(Int64(dst_i-src_i))
        dst[j] += delta
        for k in 1:signed(group_length)-1 # TODO: add a benchmark that stresses compaction and try hoisting this bounds checking
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

# Precompile
precompile(WeightBasedSampler, ())
precompile(push!, (WeightBasedSampler, Int, Float64))
precompile(delete!, (WeightBasedSampler, Int))
precompile(rand, (typeof(Random.default_rng()), WeightBasedSampler))
precompile(rand, (WeightBasedSampler,))

end
