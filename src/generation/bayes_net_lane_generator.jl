export
    BayesNetLaneGenerator,
    get_weights,
    rand!,
    get_target_vehicle_index,
    get_target_vehicle_id

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
    base_assignment_sampler::AssignmentSampler
    prop_bn::BayesNet
    prop_assignment_sampler::AssignmentSampler
    weights::Array{Float64}
    num_veh_per_lane::Int
    beh_gen::CorrelatedBehaviorGenerator
    target_veh_id::Int
    rng::MersenneTwister
    """
    Description:
        - A scene generator built on a bayes net that generates lanes 
        independently, and produces a scene specific to a target value.

    Args:
        - base_bn: for generating normal cars in the lane
        - base_assignment_sampler: samples continuous values from discrete classes
        - prop_bn: for generating importance sampled car
        - prop_assignment_sampler: samples continuous values from discrete classes
        - beh_gen: must be a correlated behavior generator, samples driver 
            given aggressiveness values
        - num_veh_per_lane: number of vehicles per lane
    """
    function BayesNetLaneGenerator(
            base_bn::BayesNet, 
            base_assignment_sampler::AssignmentSampler,
            prop_bn::BayesNet,
            prop_assignment_sampler::AssignmentSampler, 
            num_veh_per_lane::Int, 
            beh_gen::CorrelatedBehaviorGenerator, 
            rng = MersenneTwister(1)
        )
        return new(base_bn, base_assignment_sampler, prop_bn,
            prop_assignment_sampler, ones(1, num_veh_per_lane), 
            num_veh_per_lane, beh_gen, 0, rng)
    end
end

function BayesNetLaneGenerator(
        base_bn_filepath::String,
        prop_bn_filepath::String,
        num_veh_per_lane::Int,
        beh_gen::CorrelatedBehaviorGenerator,
        rng::MersenneTwister = MersenneTwister(1)
    )
    d = JLD.load(base_bn_filepath) 
    base_bn = d["bn"]
    base_sampler = AssignmentSampler(d["discs"])

    d = JLD.load(prop_bn_filepath) 
    prop_bn = d["bn"]
    prop_sampler = AssignmentSampler(d["discs"])
    return BayesNetLaneGenerator(base_bn, base_sampler, prop_bn, prop_sampler,
        num_veh_per_lane, beh_gen, rng)
end

get_weights(gen::BayesNetLaneGenerator) = gen.weights
get_target_vehicle_id(gen::BayesNetLaneGenerator) = gen.target_veh_id

function build_vehicle(
        veh_id::Int, 
        roadway::Roadway, 
        pos::Float64, 
        vel::Float64,
        vehlength::Float64,
        vehwidth::Float64,
        lane_idx::Int = 1
    )
    road_idx = RoadIndex(proj(VecSE2(0.0, 3. * (lane_idx-1), 0.0), roadway))
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, vel)
    veh_state = move_along(veh_state, roadway, pos)
    veh_def = VehicleDef(AgentClass.CAR, vehlength, vehwidth)
    return Vehicle(veh_state, veh_def, veh_id)
end

function get_target_vehicle_index(gen::BayesNetLaneGenerator, roadway::Roadway)
    target_veh_id = Int(ceil(
        floor(nlanes(roadway) / 2) 
        * gen.num_veh_per_lane 
        + gen.num_veh_per_lane - 1))
    return target_veh_id
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
    gen.target_veh_id = target_veh_id = get_target_vehicle_index(gen, roadway)
    total_num_vehicles = nlanes(roadway) * gen.num_veh_per_lane

    # generate lanes independently
    for lane_idx in 1:nlanes(roadway)
        a = Assignment()
        evidence = Assignment()
        pos = get_total_roadway_length(roadway) * 2 / 3.
        vel = Inf

        for veh_idx in 1:gen.num_veh_per_lane
            veh_id = veh_idx + (lane_idx - 1) * gen.num_veh_per_lane

            # randomly sample given the evidence and set the probability
            if veh_id == target_veh_id
                # swap evidence discretization
                prop_evidence = swap_discretization(evidence, 
                    gen.base_assignment_sampler, gen.prop_assignment_sampler)
                rand!(a, gen.prop_bn, prop_evidence)
                # note: this swaps the discretization from the proposal
                # bn to the base bn such that bins below or above the 
                # bounds of the base bn are clamped to the lowest/highest 
                # bin, which means that samples that are given 0 probability 
                # in the base bn are actually given positive probability 
                # as a result of this conversion
                base_a = swap_discretization(a, gen.prop_assignment_sampler, 
                    gen.base_assignment_sampler)
                # only the vehicle sampled from the proposal distribution 
                # will have a weight != 1
                gen.weights[veh_id] = pdf(gen.base_bn, base_a) / pdf(gen.prop_bn, a)

                # sample continuous values from the discrete assignments
                values = rand(gen.prop_assignment_sampler, a)

                # # inspect the relative probabilities of the variables in the 
                # # assignment
                # if gen.weights[veh_id] > 0.
                #     println(gen.weights[veh_id])
                #     println("values")
                #     println(values)
                #     println()
                #     for (k,v) in base_a
                #         println("key: $(k)")
                #         println("base value: $(v)")
                #         println("prop value: $(a[k])")
                #         base_cpd = get(gen.base_bn, k)
                #         prop_cpd = get(gen.prop_bn, k)

                #         println("base prob: $(pdf(base_cpd, base_a))")
                #         println("prop prob: $(pdf(prop_cpd, a))")
                #     end
                #     readline()
                # end

            else
                rand!(a, gen.base_bn, evidence)
                # sample continuous values from the discrete assignments
                values = rand(gen.base_assignment_sampler, a)
            end

            # set velocity if not yet set
            # subsequently update this velocity, discretizing it to do sampling 
            # but do not take the resampled value, instead use the true value
            # not that this means the values[:forevelocity] will be invalid
            # after the initial sample, so set inside if just for accuracy
            if vel == Inf
                vel = values[:forevelocity] + values[:relvelocity]
            else
                values[:forevelocity] = vel
                vel += values[:relvelocity]
            end
            # propagate forward the variables of this car to be evidence for 
            # the next vehicle
            evidence[:forevelocity] = encode(
                gen.base_assignment_sampler, vel, :forevelocity)

            # position should be updated as the foredistance plus half the 
            # vehicle length because that is where foredistance is 
            # measured from
            pos = pos - values[:foredistance] - values[:vehlength] / 2
            # build and add the vehicle to the scene
            veh = build_vehicle(veh_id, roadway, pos, vel, values[:vehlength],
                values[:vehwidth], lane_idx)
            # position subsequently updated to the rear of the vehicle
            pos = pos - values[:vehlength] / 2
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
            
        end
    end
    return scene
end
