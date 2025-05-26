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
# 2                      max_level::Int # absolute pointer to the last element of level weights that is nonzero

# 3                      shift::Int level weights are equal to significand_sums<<(exponent+shift), plus one if significand_sum is not zero
# 4                      sum(level weights)::UInt64
# 5..2050                level weights::[UInt64 2046] # earlier is lower. first is exponent 0x001, last is exponent 0x7fe. Subnormal are not supported (TODO).
# 2051..6142             significand_sums::[UInt128 2046] # sum of significands (the maximum significand contributes 0xfffffffffffff800)
# 6143..10234            level location info::[NamedTuple{pos::Int, length::Int} 2046] indexes into sub_weights, pos is absolute into m.
# 10235..10266           level_weights_nonzero::[Bool 2046] # map of which levels have nonzero weight (used to bump m2 efficiently when a level is zeroed out)
# 2 unused bits

# gc info:
# 10267                  next_free_space::Int (used to re-allocate)
# 16 unused bits
# 10268..10523           level allocated length::[UInt8 2046] (2^(x-1) is implied)

# 10524..10523+len       edit_map (maps index to current location in sub_weights)::[(pos<<11 + exponent)::UInt64] (zero means zero; fixed location, always at the start. Force full realloc when it OOMs. (len refers to allocated length, not m[1])

# 10524+2len..10523+7len sub_weights (woven with targets)::[[significand::UInt64, target::Int}]]. allocated_len == length_from_memory(length(m)) (len refers to allocated length, not m[1])

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
@assert 0xfffffffffffff800//big(2)^64 == (UInt64(2)^53-1)//UInt64(2)^53 == big(prevfloat(2.0))/big(2.0)
@assert 0x8000000000000000 | (reinterpret(UInt64, 1.0::Float64) << 11) === 0x8000000000000000
@assert 0x8000000000000000 | (reinterpret(UInt64, prevfloat(1.0)::Float64) << 11) === 0xfffffffffffff800
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
    x = @inline rand(rng, Random.Sampler(rng, Base.OneTo(m[5]), Val(1)))
    i = _convert(Int, m[3])
    mi = m[i]
    @inbounds while i > 6
        x <= mi && break
        x -= mi
        i -= 1
        mi = m[i]
    end

    if x >= mi # mi is the weight rounded down plus 1. If they are equal than we should refine further and possibly reject.
        # Low-probability rejection to improve accuracy from very close to perfect.
        # This branch should typically be followed with probability < 2^-21. In cases where
        # the probability is higher (i.e. m[5] < 2^32), _rand_slow_path will mutate m by
        # modifying m[4] and recomputing approximate weights to increase m[5] above 2^32.
        # This branch is still O(1) but constant factors don't matter except for in the case
        # of repeated large swings in m[5] with calls to rand interspersed.
        x > mi && error("This should be unreachable!")
        if @noinline _rand_slow_path(rng, m, i)
            @goto reject
        end
    end

    # Lookup level info
    j = 2i + 6132
    pos = m[j]
    len = m[j+1]

    # Sample within level
    while true
        r = rand(rng, UInt64)
        k1 = (r>>leading_zeros(len-1))
        k2 = _convert(Int, k1<<1+pos)
        # TODO for perf: delete the k1 < len check by maintaining all the out of bounds m[k2] equal to 0
        rand(rng, UInt64) < m[k2] * (k1 < len) && return Int(signed(m[k2+1]))
    end
end

function _rand_slow_path(rng::AbstractRNG, m::Memory{UInt64}, i)
    # shift::Int = exponent+m[4]
    # significand_sum::UInt128 = ...
    # weight::UInt64 = significand_sum<<shift+1
    # true_weight::ExactReal = exact(significand_sum)<<shift
    # true_weight::ExactReal = significand_sum<<shift + exact(significand_sum)<<shift & ...0000.1111...
    # rejection_p = weight-true_weight = (significand_sum<<shift+1) - (significand_sum<<shift + exact(significand_sum)<<shift & ...0000.1111...)
    # rejection_p = 1 - exact(significand_sum)<<shift & ...0000.1111...
    # acceptance_p = exact(significand_sum)<<shift & ...0000.1111...  (for example, if significand_sum<<shift is exact, then acceptance_p will be zero)
    # TODO for confidence: add a test that fails if this were to mix up floor+1 and ceil.
    exponent = i-5
    shift = signed(exponent + m[4])
    significand_sum = get_significand_sum(m, i)

    m4 = m[5]
    if m4 < UInt64(1)<<32
        # If the sum of approximate weights becomes less than 2^32, then for performance reasons (to keep this low probability rejection step sufficiently low probability)
        # Increase the shift to a reasonable level.
        # The fact that we are here past the isempty check in `rand` means that there are some nonzero weights.

        m2 = signed(m[3])
        x = zero(UInt64)
        checkbounds(m, 2m2-2Sys.WORD_SIZE+2041:2m2+2041)
        @inbounds for i in Sys.WORD_SIZE:-1:0 # This loop is backwards so that memory access is forwards. TODO for perf, we can get away with shaving 1 to 10 off of this loop.
            # This can underflow from significand sums into weights, but that underflow is safe because it can only happen if all the latter weights are zero. Be careful about this when re-arranging the memory layout!
            x += m[2m2-2i+2041] >> (i - 1)
        end

        # x is computed by rounding down at a certain level and then summing (and adding 1)
        # m[5] will be computed by rounding up at a more precise level and then summing
        # x could be 0 (treated as 1/2 when computing log2 with top_set_bit), composed of
        # .9 + .9 + .9 + ... for up to about log2(length) levels
        # meaning m[5] could be up to 2log2(length) times greater than predicted according to x2
        # if length is 2^64 than this could push m[5]'s top set bit up to 9 bits higher.

        # If, on the other hand, x was computed with significantly higher precision, then
        # it could overflow if there were 2^64 elements in a weight. We could probably
        # squeeze a few more bits out of this, but targeting 46 with a window of 46 to 53 is
        # plenty good enough.

        m3 = unsigned(-17 - Base.top_set_bit(x) - (m2 - 5))

        set_global_shift_increase!(m, m2, m3, m4) # TODO for perf: special case all call sites to this function to take advantage of known shift direction and/or magnitude; also try outlining

        @assert 46 <= Base.top_set_bit(m[5]) <= 53 # Could be a higher because of the rounding up, but this should never bump top set bit by more than about 8 # TODO for perf: delete
    end

    while true # TODO for confidence: move this to a separate, documented function and add unit tests.
        x = rand(rng, UInt64)
        # p_stage = significand_sum << shift & ...00000.111111...64...11110000
        shift += 64
        target = (significand_sum << shift) % UInt64
        x > target && return true
        x < target && return false
        shift >= 0 && return false
    end
end

function _getindex(m::Memory{UInt64}, i::Int)
    @boundscheck 1 <= i <= m[1] || throw(BoundsError(_FixedSizeWeights(m), i))
    j = i + 10524
    mj = m[j]
    mj == 0 && return 0.0
    pos = _convert(Int, mj >> 11)
    exponent = mj & 2047
    weight = m[pos]
    reinterpret(Float64, (exponent<<52) | (weight - 0x8000000000000000) >> 11)
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
    j = i + 10524
    if m[j] == 0
        _set_from_zero!(m, v, i)
    else
        _set_nonzero!(m, v, i)
    end
end

function _set_nonzero!(m, v, i)
    # TODO for performance: join these two operations
    _set_to_zero!(m, i)
    _set_from_zero!(m, v, i)
end

Base.@propagate_inbounds function get_significand_sum(m, i)
    i = _convert(Int, 2i+2040)
    significand_sum = UInt128(m[i]) | (UInt128(m[i+1]) << 64)
end
function update_significand_sum(m, i, delta)
    j = _convert(Int, 2i+2040)
    significand_sum = get_significand_sum(m, i) + delta
    m[j] = significand_sum % UInt64
    m[j+1] = (significand_sum >>> 64) % UInt64
    significand_sum
end

function _set_from_zero!(m::Memory, v::Float64, i::Int)
    uv = reinterpret(UInt64, v)
    j = i + 10524
    @assert m[j] == 0

    exponent = uv >> 52
    # update group total weight and total weight
    significand = 0x8000000000000000 | uv << 11
    weight_index = _convert(Int, exponent + 5)
    significand_sum = update_significand_sum(m, weight_index, significand) # Temporarily break the "weights are accurately computed" invariant
    m[2] = max(m[2], m[2weight_index+2041])

    if m[5] == 0 # if we were empty, set global shift (m[4]) so that m[5] will become ~2^40.
        m[4] = -24 - exponent

        shift = -24
        weight = _convert(UInt64, significand_sum << shift) + 1

        @assert Base.top_set_bit(weight-1) == 40 # TODO for perf: delete
        m[weight_index] = weight
        m[5] = weight
    else
        shift = signed(exponent + m[4])
        if Base.top_set_bit(significand_sum)+shift > 64
            # if this would overflow, drop shift so that it renormalizes down to 48.
            # this drops shift at least ~16 and makes the sum of weights at least ~2^48. # TODO: add an assert
            # Base.top_set_bit(significand_sum)+shift == 48
            # Base.top_set_bit(significand_sum)+signed(exponent + m[4]) == 48
            # Base.top_set_bit(significand_sum)+signed(exponent) + signed(m[4]) == 48
            # signed(m[4]) == 48 - Base.top_set_bit(significand_sum) - signed(exponent)
            m3 = 48 - Base.top_set_bit(significand_sum) - exponent
            # The "weights are accurately computed" invariant is broken for weight_index, but the "sum(weights) == m[5]" invariant still holds
            # set_global_shift_decrease! will do something wrong to weight_index, but preserve the "sum(weights) == m[5]" invariant.
            set_global_shift_decrease!(m, m3) # TODO for perf: special case all call sites to this function to take advantage of known shift direction and/or magnitude; also try outlining
            shift = signed(exponent + m3)
        end
        weight = _convert(UInt64, significand_sum << shift) + 1

        old_weight = m[weight_index]
        m[weight_index] = weight # The "weights are accurately computed" invariant is now restored
        m4 = m[5] # The "sum(weights) == m[5]" invariant is broken
        m4 -= old_weight
        m4, o = Base.add_with_overflow(m4, weight) # The "sum(weights) == m4" invariant now holds, though the computation overflows
        if o
            # If weights overflow (>2^64) then shift down by 16 bits
            m3 = m[4]-0x10
            set_global_shift_decrease!(m, m3, m4) # TODO for perf: special case all call sites to this function to take advantage of known shift direction and/or magnitude; also try outlining
            if weight_index > m[3] # if the new weight was not adjusted by set_global_shift_decrease!, then adjust it manually
                shift = signed(exponent+m3)
                new_weight = _convert(UInt64, significand_sum << shift) + 1

                @assert significand_sum != 0
                @assert m[weight_index] == weight

                m[weight_index] = new_weight
                m[5] += new_weight-weight
            end
        else
            m[5] = m4
        end
    end
    m[3] = max(m[3], weight_index) # Set after insertion because update_weights! may need to update the global shift, in which case knowing the old m[3] will help it skip checking empty levels
    level_weights_nonzero_index,level_weights_nonzero_subindex = get_level_weights_nonzero_indices(exponent)
    m[level_weights_nonzero_index] |= 0x8000000000000000 >> level_weights_nonzero_subindex

    # lookup the group by exponent and bump length
    group_length_index = _convert(Int, 5 + 3*2046 + 2exponent)
    group_pos = m[group_length_index-1]
    group_length = m[group_length_index]+1
    m[group_length_index] = group_length # setting this before compaction means that compaction will ensure there is enough space for this expanded group, but will also copy one index (16 bytes) of junk which could access past the end of m. The junk isn't an issue once coppied because we immediately overwrite it. The former (copying past the end of m) only happens if the group to be expanded is already kissing the end. In this case, it will end up at the end after compaction and be easily expanded afterwords. Consequently, we treat that case specially and bump group length and manually expand after compaction
    allocs_index,allocs_subindex = get_alloced_indices(exponent)
    allocs_chunk = m[allocs_index]
    log2_allocated_size = allocs_chunk >> allocs_subindex % UInt8 - 1
    allocated_size = 1<<log2_allocated_size

    # if there is not room in the group, shift and expand
    if group_length > allocated_size
        next_free_space = m[10268]
        # if at end already, simply extend the allocation # TODO see if removing this optimization is problematic; TODO verify the optimization is triggering
        if next_free_space == (group_pos-2)+2group_length # note that this is valid even if group_length is 1 (previously zero).
            new_allocation_length = max(2, 2allocated_size)
            new_next_free_space = next_free_space+new_allocation_length
            if new_next_free_space > length(m)+1 # There isn't room; we need to compact
                m[group_length_index] = group_length-1 # See comment above; we don't want to copy past the end of m
                next_free_space = compact!(m, m)
                group_pos = next_free_space-new_allocation_length # The group will move but remian the last group
                new_next_free_space = next_free_space+new_allocation_length
                @assert new_next_free_space < length(m)+1 # TODO for perf, delete this
                m[group_length_index] = group_length

                # Re-lookup allocated chunk because compaction could have changed other
                # chunk elements. However, the allocated size of this group could not have
                # changed because it was previously maxed out.
                allocs_chunk = m[allocs_index]
                @assert log2_allocated_size == allocs_chunk >> allocs_subindex % UInt8 - 1
                @assert allocated_size == 1<<log2_allocated_size
            end
            # expand the allocated size and bump next_free_space
            new_chunk = allocs_chunk + UInt64(1) << allocs_subindex
            m[allocs_index] = new_chunk
            m[10268] = new_next_free_space
        else # move and reallocate (this branch also handles creating new groups: TODO expirment with perf and clarity by splicing that branch out)
            twice_new_allocated_size = max(0x2,allocated_size<<2)
            new_next_free_space = next_free_space+twice_new_allocated_size
            if new_next_free_space > length(m)+1 # out of space; compact. TODO for perf, consider resizing at this time slightly eagerly?
                m[group_length_index] = group_length-1 # incrementing the group length before compaction is spotty because if the group was previously empty then this new group length will be ignored (compact! loops over sub_weights, not levels)
                next_free_space = compact!(m, m)
                m[group_length_index] = group_length
                new_next_free_space = next_free_space+twice_new_allocated_size
                @assert new_next_free_space < length(m)+1 # After compaction there should be room TODO for perf, delete this

                group_pos = m[group_length_index-1] # The group likely moved during compaction

                # Re-lookup allocated chunk because compaction could have changed other
                # chunk elements. However, the allocated size of this group could not have
                # changed because it was previously maxed out.
                allocs_chunk = m[allocs_index]
                @assert log2_allocated_size == allocs_chunk >> allocs_subindex % UInt8 - 1
                @assert allocated_size == 1<<log2_allocated_size
            end
            # TODO for perf: make compact! re-allocate the expanded group larger so there's no need to double the allocated size here if the compact branch is taken
            # TODO for perf, try removing the moveie before compaction (tricky: where to store that info?)
            # TODO make this whole alg dry, but only after setting up robust benchmarks in CI

            # expand the allocated size and bump next_free_space
            new_chunk = allocs_chunk + UInt64(1) << allocs_subindex
            m[allocs_index] = new_chunk

            m[10268] = new_next_free_space

            # Copy the group to new location
            (v"1.11" <= VERSION || 2group_length-2 != 0) && unsafe_copyto!(m, next_free_space, m, group_pos, 2group_length-2) # TODO for clarity and maybe perf: remove this version check

            # Adjust the pos entries in edit_map (bad memory order TODO: consider unzipping edit map to improve locality here)
            delta = (next_free_space-group_pos) << 11
            for k in 1:group_length-1
                target = m[_convert(Int, next_free_space)+2k-1]
                l = _convert(Int, target + 10524)
                m[l] += delta
            end

            # Mark the old group as moved so compaction will skip over it TODO: test this
            # TODO for perf: delete this and instead have compaction check if the index
            # pointed to by the start of the group points back (in the edit map) to that location
            if allocated_size != 0
                m[_convert(Int, group_pos)+1] = unsigned(Int64(-allocated_size))
            end

            # update group start location
            group_pos = m[group_length_index-1] = next_free_space
        end
    end

    # insert the element into the group
    group_lastpos = _convert(Int, (group_pos-2)+2group_length)
    m[group_lastpos] = significand
    m[group_lastpos+1] = i

    # log the insertion location in the edit map
    m[j] = _convert(UInt64, group_lastpos) << 11 + exponent

    nothing
end

function set_global_shift_increase!(m::Memory, m2, m3::UInt64, m4) # Increase shift, on deletion of elements
    @assert signed(m[4]) < signed(m3)
    m[4] = m3
    # Story:
    # In the likely case that the weight decrease resulted in a level's weight hitting zero
    # that level's weight is already updated and m4 adjusted accordingly TODO for perf don't adjust, pass the values around instead
    # In any event, m4 is accurate for current weights and all weights and significand_sums's above (before) m2 are zero so we don't need to touch them
    # Between m2 and i1, weights that were previously 1 may need to be increased. Below (past, after) i1, all weights will round up to 1 or 0 so we don't need to touch them

    #=
    weight = UInt64(significand_sum<<shift) + 1
    when is that always 1? when
    UInt64(significand_sum<<shift) == 0
    and because shift < 0 and significand_sum could be as much as 2^(Base.top_set_bit(m2))*2^64-1,
    shift <= -Base.top_set_bit(m2)-64
    shift = signed(exponent+m4)
    shift = signed(i-4+m4)
    signed(i-4+m4) <= -Base.top_set_bit(m2)-64
    i <= -signed(m4)-Base.top_set_bit(m2)-64+4
    So for -signed(m4)-60-Base.top_set_bit(m2) < i, we could need to adjust the i-th weight
    =#
    m1_next2pow = Base.top_set_bit(m[2])
    r0 = max(6, -signed(m3)-59-m1_next2pow)
    r1 = m2

    # shift = signed(i-4+m3)
    # weight = significand_sum == 0 ? 0 : UInt64(significand_sum << shift) + 1
    # shift < -64; the low 64 bits are shifted off.
    # i < -60-signed(m3); the low 64 bits are shifted off.

    checkbounds(m, r0:2r1+2041)
    @inbounds for i in r0:min(r1, -60-signed(m3))
        significand_sum_lo = m[_convert(Int, 2i+2040)]
        significand_sum_hi = m[_convert(Int, 2i+2041)]
        significand_sum_lo == significand_sum_hi == 0 && continue # in this case, the weight was and still is zero
        shift = signed(i-5+m3) + 64
        m4 += update_weight!(m, i, significand_sum_hi << shift)
    end
    @inbounds for i in max(r0,-59-signed(m3)):r1
        significand_sum = get_significand_sum(m, i)
        significand_sum == 0 && continue # in this case, the weight was and still is zero
        shift = signed(i-5+m3)
        m4 += update_weight!(m, i, significand_sum << shift)
    end

    m[5] = m4
end

function set_global_shift_decrease!(m::Memory, m3::UInt64, m4=m[5]) # Decrease shift, on insertion of elements
    m3_old = m[4]
    m[4] = m3
    @assert signed(m3) < signed(m3_old)

    # In the case of adding a giant element, call this first, then add the element.
    # In any case, this only adjusts elements at or before m[3]
    # from the first index that previously could have had a weight > 1 to min(m[3], the first index that can't have a weight > 1) (never empty), set weights to 1 or 0
    # from the first index that could have a weight > 1 to m[3] (possibly empty), shift weights by delta.
    m2 = signed(m[3])
    m1_next2pow = Base.top_set_bit(m[2])
    i1 = -signed(m3)-59-m1_next2pow # see above, this is the first index that could have weight > 1 (anything after this will have weight 1 or 0)
    i1_old = -signed(m3_old)-59-m1_next2pow # anything before this is already weight 1 or 0
    flatten_range = max(i1_old, 6):min(m2, i1-1)
    recompute_range = max(i1, 6):m2
    # From the level where one element contributes 2^64 to the level where one element contributes 1 is 64, and from there to the level where 2^64 elements contributes 1 is another 2^64.
    @assert length(flatten_range) <= 64+m1_next2pow+1
    @assert length(recompute_range) <= 64+m1_next2pow+1

    checkbounds(m, flatten_range)
    @inbounds for i in flatten_range # set nonzeros to 1
        old_weight = m[i]
        weight = old_weight != 0
        m[i] = weight
        m4 += weight-old_weight
    end

    delta = m3_old-m3
    checkbounds(m, recompute_range)
    @inbounds for i in recompute_range
        old_weight = m[i]
        old_weight <= 1 && continue # in this case, the weight was and still is 0 or 1
        m4 += update_weight!(m, i, (old_weight-1) >> delta)
    end

    m[5] = m4
end

Base.@propagate_inbounds function update_weight!(m::Memory{UInt64}, i, shifted_significand_sum)
    weight = _convert(UInt64, shifted_significand_sum) + 1
    old_weight = m[i]
    m[i] = weight
    weight-old_weight
end

get_alloced_indices(exponent::UInt64) = _convert(Int, 10269 + exponent >> 3), exponent << 3 & 0x38
get_level_weights_nonzero_indices(exponent::UInt64) = _convert(Int, 10236 + exponent >> 6), exponent & 0x3f

function _set_to_zero!(m::Memory, i::Int)
    # Find the entry's pos in the edit map table
    j = i + 10524
    mj = m[j]
    mj == 0 && return # if the entry is already zero, return
    pos = _convert(Int, mj >> 11)
    exponent = mj & 2047
    # set the entry to zero (no need to zero the exponent)
    # m[j] = 0 is moved to after we adjust the edit_map entry for the shifted element, in case there is no shifted element

    # update group total weight and total weight
    significand = m[pos]
    weight_index = _convert(Int, exponent + 5)
    significand_sum = update_significand_sum(m, weight_index, -UInt128(significand))
    old_weight = m[weight_index]
    m4 = m[5]
    m4 -= old_weight
    if significand_sum == 0 # We zeroed out a group
        level_weights_nonzero_index,level_weights_nonzero_subindex = get_level_weights_nonzero_indices(exponent)
        chunk = m[level_weights_nonzero_index] &= ~(0x8000000000000000 >> level_weights_nonzero_subindex)
        m[weight_index] = 0
        if m4 == 0 # There are no groups left
            m[3] = 5
        else
            m2 = m[3]
            if weight_index == m2 # We zeroed out the first group
                while chunk == 0 # Find the new m[3]
                    level_weights_nonzero_index -= 1
                    m2 -= 64
                    chunk = m[level_weights_nonzero_index]
                end
                m2 += 63-trailing_zeros(chunk) - level_weights_nonzero_subindex
                m[3] = m2
            end
        end
    else # We did not zero out a group
        shift = signed(exponent + m[4])
        new_weight = _convert(UInt64, significand_sum << shift) + 1
        m[weight_index] = new_weight
        m4 += new_weight
    end

    m[5] = m4 # This might be less than 2^32, but that's okay. If it is, and that's relevant, it will be corrected in _rand_slow_path

    # lookup the group by exponent
    group_length_index = _convert(Int, 5 + 3*2046 + 2exponent)
    group_pos = m[group_length_index-1]
    group_length = m[group_length_index]
    group_lastpos = _convert(Int, (group_pos-2)+2group_length)

    # TODO for perf: see if it's helpful to gate this on pos != group_lastpos
    # shift the last element of the group into the spot occupied by the removed element
    m[pos] = m[group_lastpos]
    shifted_element = m[pos+1] = m[group_lastpos+1]

    # adjust the edit map entry of the shifted element
    m[_convert(Int, shifted_element) + 10524] = _convert(UInt64, pos) << 11 + exponent
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
    # m .= 0 # This is here so that a sparse rendering for debugging is easier TODO for tests: set this to 0xdeadbeefdeadbeed
    m[5:10524+len] .= 0 # metadata and edit map need to be zeroed but the bulk does not
    m[1] = len
    m[3] = 5
    m[2] = 0
    # no need to set m[4]
    m[10268] = 10525+len
    _FixedSizeWeights(m)
end
allocated_memory(length::Integer) = 10524 + 7*length # TODO for perf: consider giving some extra constant factor allocation to avoid repeated compaction at small sizes
length_from_memory(allocated_memory::Integer) = Int((allocated_memory-10524)/7)

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
Weights, though, not just this function).
"""
function _resize!(w::ResizableWeights, len::Integer)
    m = w.m
    old_len = m[1]
    m2 = Memory{UInt64}(undef, allocated_memory(len))
    # m2 .= 0 # For debugging; TODO: set to 0xdeadbeefdeadbeef to test
    m2[1] = len
    if len > old_len # grow
        unsafe_copyto!(m2, 2, m, 2, old_len + 10524)
        m2[old_len + 10525:len + 10524] .= 0
    else # shrink
        unsafe_copyto!(m2, 2, m, 2, len + 10524)
    end

    compact!(m2, m)
    w.m = m2
    w
end

function compact!(dst::Memory{UInt64}, src::Memory{UInt64})
    dst_i = length_from_memory(length(dst)) + 10525
    src_i = length_from_memory(length(src)) + 10525
    next_free_space = src[10268]

    while src_i < next_free_space

        # Skip over abandoned groups TODO refactor these loops for clarity
        target = signed(src[src_i+1])
        while target < 0
            if unsigned(target) < 0xc000000000000000 # empty non-abandoned group; let's clean it up
                @assert 0x8000000000000001 <= unsigned(target) <= 0x80000000000007fe
                exponent = unsigned(target) - 0x8000000000000000 # TODO for clarity: dry this
                allocs_index, allocs_subindex = get_alloced_indices(exponent)
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
        j = target + 10524
        exponent = src[j] & 2047

        # Lookup the group in the group location table to find its length (performance optimization for copying, necessary to decide new allocated size and update pos)
        # exponent of 0x00000000000007fe is index 6+3*2046
        # exponent of 0x0000000000000001 is index 4+5*2046
        group_length_index = _convert(Int, 5 + 3*2046 + 2exponent)
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
        allocs_index, allocs_subindex = get_alloced_indices(exponent)
        allocs_chunk = dst[allocs_index]
        log2_allocated_size = allocs_chunk >> allocs_subindex % UInt8 - 1
        log2_new_allocated_size = group_length == 0 ? 0 : Base.top_set_bit(group_length-1)
        new_chunk = allocs_chunk + Int64(log2_new_allocated_size - log2_allocated_size) << allocs_subindex
        dst[allocs_index] = new_chunk

        # Adjust the pos entries in edit_map (bad memory order TODO: consider unzipping edit map to improve locality here)
        delta = unsigned(Int64(dst_i-src_i)) << 11
        dst[j] += delta
        for k in 1:signed(group_length)-1 # TODO: add a benchmark that stresses compaction and try hoisting this bounds checking
            target = src[src_i+2k+1]
            j = _convert(Int, target + 10524)
            dst[j] += delta
        end

        # Copy the group to a compacted location
        unsafe_copyto!(dst, dst_i, src, src_i, 2group_length)

        # Advance indices
        src_i += 2*1<<log2_allocated_size # TODO add test that fails if the 2* part is removed
        dst_i += 2*1<<log2_new_allocated_size
    end
    @label break_outer
    dst[10268] = dst_i
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
