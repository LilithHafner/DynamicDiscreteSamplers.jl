using DynamicDiscreteSamplers
using HypothesisTests
using Random
using Test
using Aqua
using Random
using StableRNGs
using StatsBase

@test DynamicDiscreteSamplers.DEBUG === true

@testset "basic end to end tests" begin
    ds = ResizableWeights(512)
    ds[1] = 1.0
    ds[2] = 2.0
    ds[3] = 4.0
    ds[1] = 0.0
    ds[2] = 0.0
    @test rand(ds) == 3
    ds[1] = 3.0
    ds[1] = 0.0
    @test rand(ds) == 3

    ds = ResizableWeights(512)
    ds[1] = 5.0
    ds[2] = 6.0
    ds[1] = 0.0
    ds[2] = 0.0
    ds = ResizableWeights(512)
    ds[1] = 1.0
    ds[2] = 2.0
    ds[2] = 0.0

    ds = ResizableWeights(512)
    for i in 1:65
        ds[i] = 2.0^i
    end
    ds[65] = 0.0
    ds[65] = 1.0
    ds[64] = 0.0
end

@testset "randomized end to end tests" begin
    ds = ResizableWeights(512)
    x = randperm(100)
    y = exp.(10*rand(100).-5);
    setindex!.((ds,), y, x)
    for _ in 1:100
        rand(ds)
    end
    for i in randperm(99)
        ds[i] = 0.0
    end
    @test rand(ds) == 100
end

@testset "interleaved randomized end to end tests" begin
    ds = ResizableWeights(512)
    elements = Set{Int}()
    for i in 1:30000
        if rand() < 0.5
            i = rand(1:10000)
            if i ∉ elements
                i > length(ds) && resize!(ds, max(i, 2length(ds)))
                ds[i] = exp(100randn())
                push!(elements, i)
            end
        elseif length(elements) > 0
            element = rand(elements)
            ds[element] = 0.0
            delete!(elements, element)
        end

        if length(elements) > 0
            @test rand(ds) in elements
        end
    end
end

@testset "Targeted statistical tests" begin
    ds = ResizableWeights(512)
    for i in 1:3
        ds[i] = float(i)
    end
    ds[2] = 0.0
    @test 0 < count(rand(ds) == 1 for _ in 1:4000) < 1200 # False positivity rate < 4e-13
end

@testset "Randomized statistical tests" begin
    rng = StableRNG(42)
    b = 100
    range = 1:b
    weights = [Float64(i) for i in range]

    ds1 = ResizableWeights(512)
    for (i, w) in zip(range, weights)
        ds1[i] = w
    end

    samples_counts = countmap([rand(rng, ds1) for _ in 1:10^5])
    counts_est = [samples_counts[i] for i in 1:b]
    ps_exact = [i/((b ÷ 2)*(b+1)) for i in 1:b]

    chisq_test = ChisqTest(counts_est, ps_exact)
    @test pvalue(chisq_test) > 0.002

    samples_counts = countmap([rand(rng, ds1) for _ in 1:10^5])
    counts_est = [samples_counts[i] for i in 1:b]
    ps_exact = [i/((b ÷ 2)*(b+1)) for i in 1:b]

    chisq_test = ChisqTest(counts_est, ps_exact)
    @test pvalue(chisq_test) > 0.002

    for i in 1:(b ÷ 2)
        ds1[i] = 0.0
    end

    samples_counts = countmap([rand(rng, ds1) for _ in 1:10^5])
    counts_est = [samples_counts[i] for i in (b ÷ 2 + 1):b]
    ps_exact = [i/((b ÷ 2)*(b+1) - (b ÷ 4)*(b ÷ 2 + 1)) for i in (b ÷ 2 + 1):b]

    chisq_test = ChisqTest(counts_est, ps_exact)
    @test pvalue(chisq_test) > 0.002

    ds2 = ResizableWeights(512)

    for (i, w) in zip(range, weights)
        ds2[i] = w
    end

    ds2[1] = 0.0
    ds2[2] = 0.0

    ds2[2] = 200.0
    resize!(ds2, 1024)
    ds2[1000] = 1000.0

    samples_counts = countmap(rand(rng, ds2, 10^5))
    counts_est = [samples_counts[i] for i in [2:b..., 1000]]
    wsum = (b ÷ 2)*(b+1) - 3 + 200 + 1000
    ps_exact = [i == 2 ? 200/wsum : i/wsum for i in [2:b..., 1000]]

    chisq_test = ChisqTest(counts_est, ps_exact)
    @test pvalue(chisq_test) > 0.002

    for i in [2:b..., 1000]
        ds2[i] = 0.0
    end

    ds2[1] = 1.0
    ds2[2] = 2.0

    samples_counts = countmap([rand(rng, ds2) for _ in 1:10^4])
    counts_est = [samples_counts[1], samples_counts[2]]
    ps_exact = [1/3, 2/3]

    chisq_test = ChisqTest(counts_est, ps_exact)
    @test pvalue(chisq_test) > 0.002

    ds2[2] = 0.0
    @test unique([rand(rng, ds2) for _ in 1:10^3]) == [1]
end


@testset "rng usage tests" begin
    function getstate_default_rng()
        t = current_task()
        (t.rngState0, t.rngState1, t.rngState2, t.rngState3, t.rngState4)
    end
    ds = ResizableWeights(512)
    ds[1] = 1.0
    state1 = getstate_default_rng()
    rand(ds)
    state2 = getstate_default_rng()
    @test state1 != state2
    rand(Xoshiro(42), ds)
    state3 = getstate_default_rng()
    @test state2 == state3
end

# These tests are too slow:
if "CI" in keys(ENV)
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DynamicDiscreteSamplers, deps_compat=false)
        Aqua.test_deps_compat(DynamicDiscreteSamplers, check_extras=false)
    end
end

@testset "stress test huge probability swings" begin
    ds = ResizableWeights(512)
    ds[1] = 1e-300
    @test rand(ds) == 1
    ds[2] = 1e300
    @test rand(ds) == 2
    ds[2] = 0.0
    @test rand(ds) == 1
end

include("weights.jl")

function error_d03fb()
    ds = ResizableWeights(512)
    for i in 1:1_500
        i > length(ds) && resize!(ds, max(i, 2length(ds)))
        ds[i] = 0.1
    end
    for i in 1:25_000
        j = rand(ds)
        ds[j] = exp(8randn())
    end
end
error_d03fb() # This threw AssertionError: 48 <= Base.top_set_bit(m[4]) <= 50 90% of the time on d03fb84d1b62272c5d6ab54c49e643af9b87201b

function error_d03fb_2(n)
    w = DynamicDiscreteSamplers.FixedSizeWeights(2^n+1);
    for i in 1:2^n-1
        w[i] = .99*.5^Base.top_set_bit(i)
    end
    w[2^n] = .99
    w[2^n+1] = 1e100
    w[2^n+1] = 0
    @test UInt64(2)^32 < w.m[3]
end
error_d03fb_2.(1:15)

ds = ResizableWeights(512)
ds[2] = 1e308
ds[2] = 0.0
ds[2] = 1e308 # This previously threw
@test rand(ds) == 2
