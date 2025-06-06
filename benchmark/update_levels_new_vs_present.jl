using Plots, Chairmarks, DynamicDiscreteSamplers
function push_v!(ds, n)
    k = ds.track_info.nvalues
    for i in k+1:k+n
        push!(ds, i, exp2(i-k)+eps())
    end
    return ds
end
function remove_v!(ds, n)
    k = ds.track_info.nvalues
    for i in k:-1:k-n+1
        delete!(ds, i)
    end
    return ds
end

x = 1:500;

y1 = [(@b push_v!(DynamicDiscreteSampler(), xi) push_v!(_, xi) evals=1 seconds=.02).time for xi in x];
y2 = [(@b DynamicDiscreteSampler() push_v!(_, xi) evals=1 seconds=.02).time for xi in x];
y3 = [(@b push_v!(push_v!(DynamicDiscreteSampler(), xi), xi) remove_v!(_, xi) evals=1 seconds=.02).time for xi in x];
y4 = [(@b push_v!(DynamicDiscreteSampler(), xi) remove_v!(_, xi) evals=1 seconds=.02).time for xi in x];

plot(x,y2,label="add element + level");
plot!(x,y1,label="add element");
plot!(x,y4,label="remove element + level");
plot!(x,y3,label="remove element")

savefig("update_levels_new_vs_present.png")
