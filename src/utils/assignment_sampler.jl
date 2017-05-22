export
    AssignmentSampler,
    rand,
    encode,
    decode,
    swap_discretization,
    get_disc_types

"""
# Description: 
    - given an assignment of variables to discrete values, this sampler 
        samples values for each of the variables according to the associated
        sampling method.

# Args:
    - discs: variable -> discretizer
    - sampling_methods: variable -> sampling method

"""
type AssignmentSampler
    discs::Dict{Symbol, AbstractDiscretizer}
    sampling_methods::Dict{Symbol, AbstractSampleMethod}
    rng::MersenneTwister
    function AssignmentSampler(
            discs::Dict{Symbol, AbstractDiscretizer},
            sampling_methods::Dict{Symbol, AbstractSampleMethod} = Dict{Symbol, AbstractSampleMethod}(),
            rng::MersenneTwister = MersenneTwister(1))
        if isempty(sampling_methods)
            for (var, disc) in discs
                if typeof(disc) == LinearDiscretizer
                    sampling_methods[var] = SAMPLE_UNIFORM
                end
            end
        end
        return new(discs, sampling_methods, rng)
    end
end

encode(samp::AssignmentSampler, v::Float64, s::Symbol) = encode(samp.discs[s], v)
get_disc_types(samp::AssignmentSampler) = Dict(
    var=>typeof(disc) for (var, disc) in samp.discs)

function rand(samp::AssignmentSampler, a::Assignment)
    values = Assignment()
    for (var, bin) in a
        if in(var, keys(samp.sampling_methods))
            values[var] = decode(samp.discs[var], bin, samp.sampling_methods[var])
        else
            values[var] = decode(samp.discs[var], bin)
        end
    end
    return values
end

function swap_discretization(a::Assignment, src::AssignmentSampler, 
        dest::AssignmentSampler)
    dest_a = Assignment()
    for (var, bin) in a
        if in(var, keys(src.sampling_methods))
            dest_a[var] = encode(dest.discs[var], 
                decode(src.discs[var], bin, src.sampling_methods[var]))
        else
            dest_a[var] = encode(dest.discs[var], decode(src.discs[var], bin))
        end
    end
    return dest_a
end
