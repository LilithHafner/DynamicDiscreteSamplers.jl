using DynamicDiscreteSamplers

t0 = time()
ds = DynamicDiscreteSampler()
push!(ds, 1, rand())
push!(ds, 2, rand())
x = rand(ds) + rand(ds)
t1 = time()

using Chairmarks

results = Dict()

results["TTFX excluding time to load"] = t1-t0

results["empty constructor"] = @be DynamicDiscreteSampler()

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
    results["constructor n=$n σ=$σ"] = @be gaussian_weights_sequential_push($n, $σ)
    results["sample n=$n σ=$σ"] = @be gaussian_weights_sequential_push(n, σ) rand
    results["delete ∘ rand n=$n σ=$σ"] = @be gaussian_weights_sequential_push(n, σ) rand_delete(_, $n) evals=1
    results["update ∘ rand n=$n σ=$σ"] = @be gaussian_weights_sequential_push(n, σ) rand_update(_, $σ) evals=n
    results["intermixed_h n=$n σ=$σ"] = @be intermixed_h($n, $σ)
    results["summarysize n=$n σ=$σ"] = Base.summarysize(gaussian_weights_sequential_push(n, σ))
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
results["pathological 1"] = @be pathological1_setup pathological1_update

function pathological2_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1e-300)
    ds
end
function pathological2_update(ds)
    push!(ds, 2, 1e300)
    delete!(ds, 2)
end
results["pathological 2"] = @be pathological2_setup pathological2_update

pathological3 = DynamicDiscreteSampler()
push!(pathological3, 1, 1e300)
delete!(pathological3, 1)
push!(pathological3, 1, 1e-300)
results["pathological 3"] = @be pathological3 rand

function pathological4_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1e-270)
    ds
end
function pathological4_update(ds)
    push!(ds, 2, 1e307)
    delete!(ds, 2)
end
results["pathological 4"] = @be pathological4_setup pathological4_update


function pathological5a_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 2.0^-32)
    push!(ds, 2, 1.0)
    ds
end
function pathological5a_update(ds)
    push!(ds, 3, 2.0^18)
    delete!(ds, 3)
end
results["pathological 5a"] = @be pathological5a_setup pathological5a_update
function pathological5b_setup()
    ds = DynamicDiscreteSampler()
    for i in 128:-1:1
        push!(ds, i, 2.0^-i)
    end
    ds
end
function pathological5b_update(ds)
    push!(ds, 129, 2.0^30)
    delete!(ds, 129)
end
results["pathological 5b"] = @be pathological5b_setup pathological5b_update

include("code_size.jl")
_code_size = code_size(dirname(pathof(DynamicDiscreteSamplers)))

results["code size in lines"] = _code_size.lines
results["code size in bytes"] = _code_size.bytes
results["code size in syntax nodes"] = _code_size.syntax_nodes

results # Return results to caller
