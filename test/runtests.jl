using DynamicDiscreteSamplers
using HypothesisTests
using Random
using Test
using Aqua
using Random
using StableRNGs
using StatsBase

@testset "unit tests" begin
    lls = DynamicDiscreteSamplers.LinkedListSet3()
    push!(lls, 2)
    push!(lls, 3)
    delete!(lls, 2)
    @test 3 in lls
end

@testset "basic end to end tests" begin
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1.0)
    push!(ds, 2, 2.0)
    push!(ds, 3, 4.0)
    delete!(ds, 1)
    delete!(ds, 2)
    @test rand(ds) == 3
    push!(ds, 1, 3.0)
    delete!(ds, 1)
    @test rand(ds) == 3

    ds = DynamicDiscreteSampler()
    push!(ds, 1, 5.0)
    push!(ds, 2, 6.0)
    delete!(ds, 1)
    delete!(ds, 2)

    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1.0)
    push!(ds, 2, 2.0)
    delete!(ds, 2)

    ds = DynamicDiscreteSampler()
    for i in 1:65
        push!(ds, i, 2.0^i)
    end
    delete!(ds, 65)
    push!(ds, 65, 1.0)
    delete!(ds, 64)
end

@testset "randomized end to end tests" begin
    ds = DynamicDiscreteSampler()
    x = randperm(100)
    y = exp.(10*rand(100).-5);
    push!.((ds,), x, y)
    for _ in 1:100
        rand(ds)
    end
    for i in randperm(99)
        delete!(ds, i)
    end
    @test rand(ds) == 100
end

@testset "interleaved randomized end to end tests" begin
    ds = DynamicDiscreteSampler()
    elements = Set{Int}()
    for i in 1:30000
        if rand() < 0.5
            i = rand(1:10000)
            if i ∉ elements
                push!(ds, i, exp(100randn()))
                push!(elements, i)
            end
        elseif length(elements) > 0
            element = rand(elements)
            delete!(ds, element)
            delete!(elements, element)
        end

        if length(elements) > 0
            @test rand(ds) in elements
        end
    end
end

@testset "Targeted statistical tests" begin
    #Issue 8
    for N in [1, 2, 4, 64, 128]
        ds = DynamicDiscreteSampler{N}()
        for i in 1:3
            push!(ds, i, float(i))
        end
        delete!(ds, 2)
        if N > 1
            @test 0 < count(rand(ds) == 1 for _ in 1:4000) < 1200 # False positivity rate < 4e-13
        else
            @test count(rand(ds) == 1 for _ in 1:4000) == 0
        end
    end
end
  
@testset "Randomized statistical tests" begin
    rng = StableRNG(42)
    b = 100
    range = 1:b
    weights = [Float64(i) for i in range]

    ds1 = DynamicDiscreteSampler()
    for (i, w) in zip(range, weights)
        push!(ds1, i, w)
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
        delete!(ds1, i)
    end

    samples_counts = countmap([rand(rng, ds1) for _ in 1:10^5])
    counts_est = [samples_counts[i] for i in (b ÷ 2 + 1):b]
    ps_exact = [i/((b ÷ 2)*(b+1) - (b ÷ 4)*(b ÷ 2 + 1)) for i in (b ÷ 2 + 1):b]

    chisq_test = ChisqTest(counts_est, ps_exact)
    @test pvalue(chisq_test) > 0.002

    ds2 = DynamicDiscreteSampler()
    
    for (i, w) in zip(range, weights)
        push!(ds2, i, w)
    end

    delete!(ds2, 1)
    delete!(ds2, 2)

    push!(ds2, 2, 200.0)
    push!(ds2, 1000, 1000.0)

    samples_counts = countmap([rand(rng, ds2) for _ in 1:10^5])
    counts_est = [samples_counts[i] for i in [2:b..., 1000]]
    wsum = (b ÷ 2)*(b+1) - 3 + 200 + 1000
    ps_exact = [i == 2 ? 200/wsum : i/wsum for i in [2:b..., 1000]]

    chisq_test = ChisqTest(counts_est, ps_exact)
    @test pvalue(chisq_test) > 0.002

    for i in [2:b..., 1000]
        delete!(ds2, i)
    end

    push!(ds2, 1, 1.0)
    push!(ds2, 2, 2.0)
    samples_counts = countmap([rand(rng, ds2) for _ in 1:10^4])
    counts_est = [samples_counts[1], samples_counts[2]]
    ps_exact = [1/3, 2/3]
    @test pvalue(chisq_test) > 0.002

    delete!(ds2, 2)
    @test unique([rand(rng, ds2) for _ in 1:10^3]) == [1]
end

# These tests are too slow:
if "CI" in keys(ENV)
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DynamicDiscreteSamplers, deps_compat=false)
        Aqua.test_deps_compat(DynamicDiscreteSamplers, check_extras=false)
    end
end
