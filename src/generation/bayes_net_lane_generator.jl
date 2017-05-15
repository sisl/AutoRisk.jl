export
    BayesNetLaneGenerator,
    get_weights,
    rand!

# forward sampling a BN with evidence generally doesn't work, since it 
# requires running inference to get the posterior
# but if we know that the evidence is in a root node (i.e., node without 
# parents), then we can easily sample from the posterior.
function Base.rand!(a::Assignment, bn::BayesNet, evidence::Assignment)
    # check that the evidence variables do not have parents, or that if they 
    # do have parents that they are all also in the evidence
    evidence_vars = keys(evidence)
    for var in evidence_vars
        @assert length(setdiff(parents(get(bn, var)), evidence_vars)) == 0
    end

    # transfer evidence to assignment
    for (k,v) in evidence
        a[k] = v
    end

    # sample as usual, skipping the evidence
    cpds = [cpd for cpd in bn.cpds if !in(BayesNets.name(cpd), evidence_vars)]
    for cpd in cpds
        a[BayesNets.name(cpd)] = rand(cpd, a)
    end
    return a
end

type BayesNetLaneGenerator <: Generator
    base_bn::BayesNet
    prop_bn::BayesNet
    assignment_sampler::AssignmentSampler
    weights::Array{Float64}
    num_veh_per_lane::Int
    beh_gen::CorrelatedBehaviorGenerator
    rng::MersenneTwister
    """
    Description:
        - A scene generator built on a bayes net that generates lanes 
        independently, and produces a scene specific to a target value.

    Args:
        - base_bn: for generating normal cars in the lane
        - prop_bn: for generating importance sampled car
        - assignment_sampler: samples continuous values from discrete classes
        - beh_gen: must be a correlated behavior generator, samples driver 
            given aggressiveness values
        - num_veh_per_lane: number of vehicles per lane
    """
    function BayesNetLaneGenerator(base_bn::BayesNet, prop_bn::BayesNet,
            assignment_sampler::AssignmentSampler, num_veh_per_lane::Int, 
            beh_gen::CorrelatedBehaviorGenerator, rng = MersenneTwister(1))
        return new(base_bn, prop_bn, assignment_sampler, 
            ones(1, num_veh_per_lane), num_veh_per_lane, beh_gen, rng)
    end
end

# propagate values for use in generating the next vehicle
function update!(dest::Assignment, gen::BayesNetLaneGenerator, src::Assignment)
    velocity = src[:relvelocity] + src[:forevelocity]
    dest[:forevelocity] = discretize(
        gen.assignment_sampler, velocity, :forevelocity)
end

update_pos(pos::Float64, a::Assignment) = pos - a[:foredistance]
get_weights(gen::BayesNetLaneGenerator) = gen.weights

function build_vehicle(veh_id::Int, a::Assignment, roadway::Roadway, 
        pos::Float64, lane_idx::Int = 1)
    road_idx = RoadIndex(proj(VecSE2(0.0, 3. * (lane_idx-1), 0.0), roadway))
    velocity = a[:relvelocity] + a[:forevelocity]
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, velocity)
    veh_state = move_along(veh_state, roadway, pos)
    veh_def = VehicleDef(AgentClass.CAR, a[:vehlength], a[:vehwidth])
    return Vehicle(veh_state, veh_def, veh_id)
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
function Base.rand!(gen::BayesNetLaneGenerator, roadway::Roadway, scene::Scene, 
        models::Dict{Int, DriverModel}, seed::Int64) 
    # set random seed
    srand(seed)
    srand(gen.rng, seed)
    srand(gen.beh_gen.rng, seed)
    empty!(models)
    empty!(scene)

    # reallocate weight vector if it is different
    total_num_veh = nlanes(roadway) * gen.num_veh_per_lane
    if total_num_veh != length(gen.weights)
        gen.weights = ones(1, total_num_veh)
    end

    # assume that the vehicle with elevated target probability is the one 
    # in the middle of the lanes, and is the second to last vehicle in its lane
    target_veh_id = Int(ceil(
        floor(nlanes(roadway) / 2) 
        * gen.num_veh_per_lane 
        + gen.num_veh_per_lane - 1))
    total_num_vehicles = nlanes(roadway) * gen.num_veh_per_lane

    # generate lanes independently
    for lane_idx in 1:nlanes(roadway)
        a = Assignment()
        evidence = Assignment()
        pos = get_total_roadway_length(roadway) * 2 / 3.

        for veh_idx in 1:gen.num_veh_per_lane
            veh_id = veh_idx + (lane_idx - 1) * gen.num_veh_per_lane

            # randomly sample given the evidence and set the probability
            if veh_id == target_veh_id
                rand!(a, gen.prop_bn, evidence)
                # only the vehicle sampled from the proposal distribution 
                # will have a weight != 1
                gen.weights[veh_id] = pdf(gen.base_bn, a) / pdf(gen.prop_bn, a)
            else
                rand!(a, gen.base_bn, evidence)
            end

            # sample continuous values from the discrete assignments
            values = rand(gen.assignment_sampler, a)

            # build and add the vehicle to the scene and models
            pos = update_pos(pos, values)
            veh = build_vehicle(veh_id, values, roadway, pos, lane_idx)
            push!(scene, veh)
            params = rand(gen.beh_gen, values[:aggressiveness])
            # if first vehicle in lane, set desired speed to current speed 
            # as a hack to prevent universal acceleration
            if veh_idx == 1
                params.idm.v_des = veh.state.v
            end
            models[veh_id] = build_driver(params, total_num_vehicles)
            srand(models[veh_id], seed + Int(rand(gen.rng, 1:1e8)))
            if typeof(models[veh_id]) == ErrorableDriverModel
                is_attentive = values[:isattentive] == 1 ? false : true
                set_is_attentive!(models[veh_id], is_attentive)
            end

            # propagate forward the variables of this car to be evidence for 
            # the next vehicle
            update!(evidence, gen, values)
        end
    end
    return scene
end
