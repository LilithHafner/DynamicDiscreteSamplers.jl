using DynamicDiscreteSamplers

t0 = time()
ds = DynamicDiscreteSampler()
push!(ds, 1, rand())
push!(ds, 2, rand())
x = rand(ds) + rand(ds)
t1 = time()

using ChairmarksForAirspeedVelocity

"Convert a constant into a format that AirspeedVelocity can understand"
constant(n) = ChairmarksForAirspeedVelocity.Runnable(Returns(ChairmarksForAirspeedVelocity.BenchmarkTools.Trial(ChairmarksForAirspeedVelocity.BenchmarkTools.Parameters(seconds=0,samples=2,evals=1,overhead=0,gctrial=false,gcsample=false),1e9Float64[n,n],Float64[0,0],0,0)))

# SUITE is a magic global variable that AirspeedVelocity looks for
SUITE = BenchmarkGroup()

SUITE["TTFX"] = constant(t1-t0)

SUITE["empty constructor"] = @benchmarkable DynamicDiscreteSampler()

function gaussian_weights_sequential_push(n, σ)
    ds = DynamicDiscreteSampler()
    for i in 1:n
        push!(ds, i, exp(σ*randn()))
    end
    ds
end

function rand_delete(ds, n)
    res = 0
    for i in 1:n
        j = rand(ds)
        delete!(ds, j)
        res = res >> 1 + j
    end
    res
end

function rand_update(ds, σ)
    j = rand(ds)
    delete!(ds, j)
    push!(ds, j, exp(σ*randn()))
    j
end

function intermixed_h(n, σ)
    ds = DynamicDiscreteSampler()
    elements = Set{Int}()
    res = 0
    for i in 1:n
        if rand() < 0.5
            element = rand(1:n)
            if element ∉ elements
                push!(ds, element, exp(σ*randn()))
                push!(elements, element)
            end
        elseif length(elements) > 0
            element = rand(elements)
            delete!(ds, element)
            delete!(elements, element)
        end
        if length(elements) > 0
            res += rand(ds)
        end
    end
    res
end

for n in [100, 1000, 10000], σ in [.1, 1, 10, 100]
    # TODO: try to use min over noise, average over rng, and max over treatment in analysis
    SUITE["constructor n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push($n, $σ)
    SUITE["sample n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push(n, σ) rand
    SUITE["delete ∘ rand n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push(n, σ) rand_delete(_, $n) evals=1
    SUITE["update ∘ rand n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push(n, σ) rand_update(_, $σ) evals=n
    SUITE["intermixed_h n=$n σ=$σ"] = @benchmarkable intermixed_h($n, $σ)
end

function pathological1_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1e50)
    ds
end
function pathological1_update(ds)
    push!(ds, 2, 1e100)
    delete!(ds, 2)
end
SUITE["pathological 1"] = @benchmarkable pathological1_setup pathological1_update

function pathological2_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1e-300)
    ds
end
function pathological2_update(ds)
    push!(ds, 2, 1e300)
    delete!(ds, 2)
end
SUITE["pathological 2"] = @benchmarkable pathological2_setup pathological2_update

pathological3 = DynamicDiscreteSampler()
push!(pathological3, 1, 1e300)
delete!(pathological3, 1)
push!(pathological3, 1, 1e-300)
SUITE["pathological 3"] = @benchmarkable pathological3 rand

function pathological4_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1e-270)
    ds
end
function pathological4_update(ds)
    push!(ds, 2, 1e307)
    delete!(ds, 2)
end
SUITE["pathological 4"] = @benchmarkable pathological4_setup pathological4_update

include("code_size.jl")
_code_size = code_size(pathof(DynamicDiscreteSamplers))

SUITE["code size in lines"] = constant(3600_code_size.lines)
SUITE["code size in bytes"] = constant(3600_code_size.bytes)
SUITE["code size in syntax nodes"] = constant(3600_code_size.syntax_nodes)
