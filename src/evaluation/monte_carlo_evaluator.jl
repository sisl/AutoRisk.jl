export 
    Evaluator,
    MonteCarloEvaluator,
    evaluate!,
    get_veh_id_to_idx,
    get_features,
    get_targets

abstract Evaluator

# default performs no bootstrapping 
function bootstrap_targets!(eval::Evaluator, models::Dict{Int, DriverModel},
        roadway::Roadway)
    return eval.targets
end

# the indexing here is to ensure that the dimension is not dropped
get_features(eval::Evaluator) = eval.features[:, 1:eval.feature_timesteps, :]
get_targets(eval::Evaluator) = eval.agg_targets[:, :]

"""
# Description:
    - MonteCarloEvaluator evaluates a set of {roadway, scene, models}
        by simulating them together many times and deriving features and 
        targets from the results.
"""
type MonteCarloEvaluator <: Evaluator
    ext::AbstractFeatureExtractor
    num_runs::Int64
    context::ActionContext
    prime_time::Float64
    sampling_time::Float64
    veh_idx_can_change::Bool

    rec::SceneRecord
    features::Array{Float64}
    feature_timesteps::Int64
    targets::Array{Float64}
    agg_targets::Array{Float64}

    rng::MersenneTwister
    num_veh::Int64
    veh_id_to_idx::Dict{Int64, Int64}
    """
    # Args:
        - ext: feature and target extractor
        - num_runs: how many monte carlo runs to run
        - context: context in which to run
        - prime_time: "burn-in" time for the scene
        - sampling_time: time to sample the scene after burn-in
        - veh_idx_can_change: whether or not the vehicle indices in the scene
            can change over time
        - rec: record to use for storage of scenes
        - features: array in which to store features, 
            shape = (feature_dim, max_num_veh)
        - targets: array in which to store targets for each monte carlo run,
            shape = (target_dim, max_num_veh)
        - agg_targets: aggregate target values accumulated across runs
        - rng: random number generator to use
    """
    function MonteCarloEvaluator(ext::AbstractFeatureExtractor,
            num_runs::Int64, 
            context::ActionContext,
            prime_time::Float64, 
            sampling_time::Float64, 
            veh_idx_can_change::Bool, 
            rec::SceneRecord, 
            features::Array{Float64}, 
            targets::Array{Float64},
            agg_targets::Array{Float64}, 
            rng::MersenneTwister = MersenneTwister(1))
        features_size = size(features)
        @assert length(features_size) == 3
        feature_timesteps = features_size[2]

        return new(ext, num_runs, context, prime_time, sampling_time, 
            veh_idx_can_change, rec, features, feature_timesteps, targets, 
            agg_targets, rng, 0, Dict{Int64, Int64}())
    end
end

"""
# Description:
    - Populate the dict with mappings of id to index in the scene

# Args:
    - scene: the scene to extract ids/indices from
    - dict: dictionary to populate
"""
function get_veh_id_to_idx(scene::Scene, dict::Dict{Int64, Int64})
    for veh in scene
        vehicle_index = get_index_of_first_vehicle_with_id(
            scene, veh.def.id)
        dict[veh.def.id] = vehicle_index
    end
end

"""
# Description:
    - Resets members of eval prior to evaluation of a scene.

# Args:
    - eval: the evaluator to reset
    - scene: the scene on which to reset
    - seed: random seed with which to reset the evaluator
"""
function reset!(eval::Evaluator, scene::Scene, seed::Int64)
    srand(seed)
    srand(eval.rng, seed)
    fill!(eval.agg_targets, 0)
    eval.num_veh = length(scene)
    empty!(eval.veh_id_to_idx)
end

"""
# Description:
    - Evaluate a {roadway, scene, models} tuple.

# Args:
    - eval: evaluator to use
    - scene: contains the vehicles to evaluate
    - models: contains driver models to use in propagating scene
    - roadway: roadway on which scene is based
    - seed: random seed used for evaluation 
"""
function evaluate!(eval::Evaluator, scene::Scene, 
        models::Dict{Int, DriverModel}, roadway::Roadway, seed::Int64)
    # reset values across this set of monte carlo runs
    reset!(eval, scene, seed)
    
    # prime the scene by simulating for short period
    # extract prediction features at this point
    simulate!(scene, models, roadway, eval.rec, eval.prime_time)

    # need this dictionary because cars may enter or exit the 
    # scene. As a result, the indices of the scene may or may 
    # not correspond to the correct vehicle ids at the end of 
    # each monte carlo run. Note that this must be performed 
    # _after_ the burn-in period since vehicles may leave 
    # the scene during that process.
    get_veh_id_to_idx(scene, eval.veh_id_to_idx)

    # extract features for all vehicles using scenes simulated so far
    pull_features!(eval.ext, eval.rec, roadway, models, eval.features, 
        eval.feature_timesteps)
    
    # repeatedly simulate, starting from the final burn-in scene 
    temp_scene = Scene(length(scene.vehicles))
    pastframe = 0 # first iteration, don't alter record
    for idx in 1:eval.num_runs
        # reset
        copy!(temp_scene, scene)
        push_forward_records!(eval.rec, -pastframe)

        # simulate starting from the final burn-in scene
        simulate!(temp_scene, models, roadway, eval.rec, eval.sampling_time)

        # pastframe is the number of frames that have been simulated
        pastframe = Int(round(eval.sampling_time / eval.rec.timestep))

        # get the initial extraction frame, this will typically be the first 
        # frame following the prime time, but in the case where no time is 
        # simulated, it should be the most recent frame
        start_extract_frame = max(pastframe - 1, 0)

        # extract target values from every frame in the record for every vehicle
        extract_targets!(eval.rec, roadway, eval.targets, eval.veh_id_to_idx,
            eval.veh_idx_can_change, start_extract_frame)

        # optionally bootstrap target values
        bootstrap_targets!(eval, models, roadway)

        # add targets to aggregate targets
        eval.agg_targets[:, 1:eval.num_veh] += eval.targets[:, 1:eval.num_veh]  
    end

    ## compute confidence intervals for each target
    # for t in 1:size(eval.agg_targets, 1)
    #     X = eval.agg_targets[t, :]
    #     N = eval.num_runs
    #     σ = sqrt(X / N - (X / N).^2) / N
    #     X /= N
    #     z = 1.96
    #     println("σ: $(σ)")
    #     ci_l = X - z * σ / sqrt(N)
    #     ci_h = X + z * σ / sqrt(N)
    #     for (i, (l, h)) in enumerate(zip(ci_l, ci_h))
    #         println("i: $(i) t: $(t) low: $(l) high: $(h)")
    #     end
    # end

    # divide by num_runs to get average values
    eval.agg_targets[:] /= eval.num_runs
end


