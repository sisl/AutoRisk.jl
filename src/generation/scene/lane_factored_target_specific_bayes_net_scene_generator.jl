
# forward sampling a BN with evidence generally doesn't work, since it 
# requires running inference to get the posterior
# but if we know that the evidence is in a root node (i.e., node without 
# parents), then we can easily sample from the posterior.
function Random.rand!(a::Assignment, bn::BayesNet, sampler::DirectSampler, 
        evidence::Assignment)
    # check that the evidence variables do not have parents, or that if they 
    # do have parents that they are all also in the evidence
    evidence_vars = keys(evidence)
    for var in evidence_vars
        @assert length(setdiff(parents(get(bn, var)), evidence_vars)) == 0
    end

    # transfer evidence to assignment
    a = union(a, evidence)

    # sample as usual, skipping the evidence
    cpds = [cpd for cpd in bn.cpds if !in(name(cpd), evidence_vars)]
    for cpd in cpds
        a[name(cpd)] = rand(cpd, a)
    end
    a
end

mutable struct LaneFactoredTargetSpecificBayesNetSceneGenerator <: SceneGenerator
    bn::BayesNet
    sampler::BayesNetSampler
    target_evidence::Dict{Symbol,Int}
    dynamics::Dict{Symbol,Symbol}
    num_veh_per_lane::Int64
    weights::Array{Float64}
    rng::MersenneTwister
    """
    Description:
        - A scene generator built on a bayes net that generates lanes 
        independently, and produces a scene specific to a target value.

    Args:
        - bn: for generating lanes
        - sampler: sampler for the bn
        - evidence: dictionary mapping between a symbol in the bn and the class
            that that symbol should assume. This should be used to specify that 
            a target(s) should be sampled at the given value(s) for a _single_ 
            vehicle in the scene. The reason only a single vehicle should ever
            be oversampled at the evidence is that if more than one vehicle is 
            oversampled then the network no longer gives accurate weights for 
            overall assignments because it factors across the individual 
            vehicles (given the leading vehicles).
        - dynamics: map from variables at time t to variables at time t+1 that 
            should assume the same values
        - num_veh_per_lane: number of vehicles in each lane
        - weights: the probability of each of the vehicle state assignments resulting 
            from calling rand (again, factors across lanes, and vehicles 
            conditioned on lead vehicle)
    """
    function LaneFactoredTargetSpecificBayesNetSceneGenerator(bn::BayesNet, 
            sampler::BayesNetSampler, evidence::Dict{Symbol,Int}, 
            dynamics::Dict{Symbol,Symbol}, num_veh_per_lane::Int64, 
            rng = MersenneTwister(1))
        return new(bn, sampler, evidence, num_veh_per_lane, 
            ones(num_veh_per_lane), rng)
    end
end

update_remaining_length(L::Float64, a::Assignment) = L - a[:foredistance]

# dynamics gives mapping of src keys to dest keys
function update!(dynamics::Dict{Symbol:Symbol}, src::Dict{Symbol,Int}, 
        dest::Dict{Symbol,Int})
    for (src_key, dest_key) in dynamics
        dest[dest_key] = src[src_key]
    end
    return dest
end

# assumes one-to-one mapping of keys from src to dest
function update!(src::Dict{Symbol,Int}, dest::Dict{Symbol,Int})
    for (src_key, src_val) in src
        dest[src_key] = src_val
    end
    return dest
end

"""
# Description:
    - Reset a scene using the generator.

# Args:
    - gen: generator to use
    - scene: scene to populate with vehicles.
    - roadway: on which to place vehicles
    - seed: random seed to use for generation
"""
function Random.rand!(gen::LaneFactoredTargetSpecificBayesNetSceneGenerator, 
        scene::Scene, roadway::Roadway, seed::Int64) 
    # set random seed
    Random.seed!(gen.rng, seed)

    # reallocate weight vector if it is different
    total_num_veh = nlanes(roadway) * gen.num_veh_per_lane
    if total_num_veh != length(gen.weights)
        gen.weights = ones(total_num_veh)
    end

    # assume that the vehicle with elevated target probability is the one 
    # in the middle of the lanes, and in the middle of the vehicles
    target_veh_id = Int(ceil(floor(nlanes(roadway) / 2) * gen.num_veh_per_lane 
        + gen.num_veh_per_lane / 2))

    # generate lanes independently
    for lane_idx in 1:nlanes(roadway)
        a = Assignment()
        evidence = Assignment()
        L = get_total_roadway_length(roadway) * 2. / 3.
        for veh_idx in 1:gen.num_veh_per_lane
            veh_id = veh_idx + (lane_idx - 1) * gen.num_veh_per_lane

            # if this is the target vehicle then deterministically set it 
            # to have the targets as specified by target_evidence
            if veh_id == target_veh_id
                update!(gen.target_evidence, a)
            end

            # propagate forward the variables from the previous car position
            update!(gen.dynamics, a, evidence)

            # randomly sample given the evidence
            rand!(a, gen.bn, gen.sampler, evidence)

            # set the probability of this assignment
            gen.weights[veh_id] = pdf(bn, a)

            # build and add the vehicle
            L = update_remaining_length(L, a)
            veh = build_vehicle(veh_id, a, roadway, L)
            push!(scene, veh)
        end
    end
    return scene
end
