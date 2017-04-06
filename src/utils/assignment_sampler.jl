export
    AssignmentSampler,
    UniformAssignmentSampler,
    random_uniform,
    rand

abstract AssignmentSampler
rand(samp::AssignmentSampler, a::Assignment) = error("not implemented")

"""
# Description: 
    - given an assignment of variables to discrete values, this sampler 
        samples random uniform values from the bins associated with each 
        discrete class

# Args:
    - var_edges: dictionary mapping symbols (the variable) to a list of cutpoints
        s.t. each discrete class in an assignment is associated with one of the 
        bins defined by these cutpoints

"""
type UniformAssignmentSampler <: AssignmentSampler
    var_edges::Dict{Symbol, Vector{Float64}}
    rng::MersenneTwister
    function UniformAssignmentSampler(var_edges::Dict{Symbol, Vector{Float64}},
            rng::MersenneTwister=MersenneTwister(1))
        return new(var_edges, rng)
    end
end

function random_uniform(rng::MersenneTwister, lo::Float64, hi::Float64)
    return lo + (hi - lo) * rand(rng)
end

function rand(samp::UniformAssignmentSampler, a::Assignment)
    # sample the variables from the appropriate uniform distribution
    values = Assignment()
    unsampled_vars = setdiff(a, keys(samp.var_edges))
    for (var, edges) in samp.var_edges
        values[var] = random_uniform(samp.rng, edges[a[var]], edges[a[var] + 1])
    end

    # if variables in the assignment are not accounted for in the sampler, 
    # then they assume the same values as those already in the assignment
    unsampled_vars = setdiff(keys(a), keys(samp.var_edges))
    for var in unsampled_vars
        values[var] = a[var]
    end

    return values
end