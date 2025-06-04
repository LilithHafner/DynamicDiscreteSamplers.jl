
using AliasTables

Random.rand!(rng::AbstractRNG, A::AbstractArray, st::Random.SamplerTrivial{<:Weights}) = _rand!(rng, A, st[].m)

function _rand!(rng::AbstractRNG, samples::AbstractArray, m::Memory{UInt64})
    n = length(samples)
    n < 100 && return fill_samples!(rng, m, samples)
    max_i = _convert(Int, m[2])
    min_i = 5
    min_j = 10495
    for j in 10495:10527
        chunk = m[j]
        if chunk != 0
            min_j = j
            min_i = (j-10495) << 6 + leading_zeros(chunk) + 4
            break
        end
    end
    k = sum(count_ones(m[j]) for j in min_j:10527)
    n < 100*(k^0.72) && return fill_samples!(rng, m, samples)
    inds = Vector{Int}(undef, k)
    weights = Vector{BigInt}(undef, k)
    l = 0
    for j in max_i:-1:min_i
        if m[j] != 0
            l += 1
            inds[l] = j
            weights[l] = BigInt(get_significand_sum(m, j)) << (j-min_i)
        end
    end
    counts = multinomial_sample(rng, n, weights)
    ct, s = 0, 1
    for i in 1:k
        c = counts[i]
        c == 0 && continue
        j = 2*inds[i] + 6289
        pos = m[j]
        len = m[j+1]
        for _ in 1:c
            samples[s] = sample_within_level(rng, m, pos, len)
            s += 1
        end
        ct += c
        ct == n && break
    end
    faster_shuffle!(rng, samples)
end

# it uses some internals from Base.GMP.MPZ (as MutableArithmetics.jl) to speed-up some BigInt operations
"""
    binomial_sample(rng, trials, px, py)
    
Flip a coin with probability of `px//py` of coming up heads `trials` times and return the number of heads.

Has `O(trials)` expected runtime with a very low constant factor.

Implementation based on Farach-Colton, M. and Tsai, M.T., 2015. Exact sublinear binomial sampling.
"""
function binomial_sample(rng, trials, px, py)
    if iszero(trials) || iszero(px)
        return 0
    elseif px == py
        return trials
    end
    count = 0
    while trials > 0
        c = binomial_sample_fair_coin(rng, trials)
        Base.GMP.MPZ.mul_2exp!(px, 1) # px *= 2
        if px > py
            count += c
            trials -= c
            Base.GMP.MPZ.sub!(px, py) # px -= py
        elseif px < py
            trials = c
        else
            count += c
            break
        end
    end
    count
end

# These are exact because their weights sum to powers of 2.
const FLIP_64_COINS = AliasTable(binomial.(BigInt(64),0:64))
const FLIP_16_COINS= AliasTable(binomial.(BigInt(16),0:16))

"""
    binomial_sample_fair_coin(rng, trials)
    
Flips `trials` fair coins and reports the number of heads.

Flips up to 64 coins at a time.
"""
function binomial_sample_fair_coin(rng, trials)
    sum((rand(rng, FLIP_64_COINS)-1 for _ in 1:trials >> 6), init=0) + 
    sum((rand(rng, FLIP_16_COINS)-1 for _ in 1:(trials >> 4) % 4), init=0) + 
    count(rand(Bool) for _ in 1:trials%16)
end

"""
    multinomial_sample(rng, trials, weights)
    
Draw `trials` elements from the probability distribution specified by `weights` (need not sum to 1) and return the number of times each element was drawn.

Runs in `O(trials * weights)`, but can be as fast as `O(trials)` if the weights are skewed toward big weights at the beginning.
"""
function multinomial_sample(rng, trials, weights::AbstractVector{<:Integer})
    sum_weights = sum(weights)
    counts = Vector{Int}(undef, length(weights))
    weight_copy = BigInt(0)
    for i in 1:length(weights)-1
        weight = weights[i]
        Base.GMP.MPZ.set!(weight_copy, weight)
        b = binomial_sample(rng, trials, weight_copy, sum_weights)
        counts[i] = b
        trials -= b
        trials == 0 && return counts
        Base.GMP.MPZ.sub!(sum_weights, weight)
    end
    counts[end] = trials
    counts
end

function fill_samples!(rng, m, samples)
    for i in eachindex(samples)
        samples[i] = _rand(rng, m)
    end
    samples
end

function faster_shuffle!(rng::AbstractRNG, vec::AbstractArray)
    for i in 2:length(vec)
        endi = _convert(UInt, i-1)
        j = @inline rand(rng, Random.Sampler(rng, UInt(0):endi, Val(1))) % Int + 1
        vec[i], vec[j] = vec[j], vec[i]
    end
    vec
end
