using AutoRisk

function build_dataset_collector(output_filepath, flags, col_id = 0)
    target_dim = flags["target_dim"]
    chunk_dim = flags["chunk_dim"]
    roadway_type = flags["roadway_type"]
    roadway_length = flags["roadway_length"]
    roadway_radius = flags["roadway_radius"]
    min_num_veh = flags["min_num_vehicles"]
    max_num_veh = flags["max_num_vehicles"]
    min_base_speed = flags["min_base_speed"]
    max_base_speed = flags["max_base_speed"]
    min_vehicle_length = flags["min_vehicle_length"]
    max_vehicle_length = flags["max_vehicle_length"]
    min_vehicle_width = flags["min_vehicle_width"]
    max_vehicle_width = flags["max_vehicle_width"]
    min_init_dist = flags["min_init_dist"]
    num_lanes = flags["num_lanes"]
    prime_time = flags["prime_time"]
    sampling_time = flags["sampling_time"]
    sampling_period = flags["sampling_period"]
    num_runs = flags["num_monte_carlo_runs"]
    veh_idx_can_change = flags["veh_idx_can_change"]
    max_num_samples = flags["num_scenarios"] * max_num_veh
    behavior_type = flags["behavior_type"]
    heuristic_behavior_type = flags["heuristic_behavior_type"]
    lon_accel_std_dev = flags["lon_accel_std_dev"]
    lat_accel_std_dev = flags["lat_accel_std_dev"]
    response_time = flags["response_time"]
    evaluator_type = flags["evaluator_type"]
    prediction_model_type = flags["prediction_model_type"]
    network_filepath = flags["network_filepath"]
    driver_network_filepath = flags["driver_network_filepath"]
    extractor_type = flags["extractor_type"]
    feature_timesteps = flags["feature_timesteps"]

    # feature_dim depends on extractor_type, so build extractor first
    if extractor_type == "heuristic"
        ext = MultiFeatureExtractor()
    elseif extractor_type == "multi"
        subexts = []
        if flags["extract_core"]
            push!(subexts, CoreFeatureExtractor())
        end
        if flags["extract_temporal"]
            push!(subexts, TemporalFeatureExtractor())
        end
        if flags["extract_well_behaved"]
            push!(subexts, WellBehavedFeatureExtractor())
        end
        if flags["extract_neighbor"]
            push!(subexts, NeighborFeatureExtractor())
        end
        if flags["extract_car_lidar"]
            push!(subexts, 
                CarLidarFeatureExtractor(extract_carlidar_rangerate = 
                    flags["extract_car_lidar_range_rate"]))
        end
        if flags["extract_road_lidar"]
            push!(subexts, RoadLidarFeatureExtractor())
        end
        ext = MultiFeatureExtractor(subexts)
    else
        throw(ArgumentError(
            "invalid extractor_type $(extractor_type)"))
    end
    feature_dim = length(ext)

    # seeds are replaced by parallel collector
    seeds = Vector{Int}()

    # roadway gen
    if roadway_type == "straight"
        roadway = gen_straight_roadway(num_lanes, roadway_length)
    else
        roadway = gen_stadium_roadway(num_lanes, length = roadway_length, 
            radius = roadway_radius)
    end
    roadway_gen = StaticRoadwayGenerator(roadway)

    # scene gen
    scene = Scene(max_num_veh)
    scene_gen = HeuristicSceneGenerator(
        min_num_veh, 
        max_num_veh, 
        min_base_speed,
        max_base_speed,
        min_vehicle_length,
        max_vehicle_length,
        min_vehicle_width, 
        max_vehicle_width,
        min_init_dist)

    # context 
    context = IntegratedContinuous(sampling_period, 1)

    if behavior_type == "heuristic"
        if heuristic_behavior_type == "aggressive"
            params = [get_aggressive_behavior_params(
                lon_σ = lon_accel_std_dev, 
                lat_σ = lat_accel_std_dev, 
                response_time = response_time)]
            weights = WeightVec([1.])
        elseif heuristic_behavior_type == "passive"
            params = [get_passive_behavior_params(
                lon_σ = lon_accel_std_dev, 
                lat_σ = lat_accel_std_dev, 
                response_time = response_time)]
            weights = WeightVec([1.])
        elseif heuristic_behavior_type == "normal"
            params = [get_normal_behavior_params(
                lon_σ = lon_accel_std_dev, 
                lat_σ = lat_accel_std_dev, 
                response_time = response_time)]
            weights = WeightVec([1.])
        else
            params = [get_aggressive_behavior_params(
                        lon_σ = lon_accel_std_dev, 
                        lat_σ = lat_accel_std_dev, 
                        response_time = response_time), 
                    get_passive_behavior_params(
                        lon_σ = lon_accel_std_dev, 
                        lat_σ = lat_accel_std_dev, 
                        response_time = response_time),
                    get_normal_behavior_params(
                        lon_σ = lon_accel_std_dev, 
                        lat_σ = lat_accel_std_dev, 
                        response_time = response_time)]
            weights = WeightVec([.2,.3,.5])
        end
        behavior_gen = PredefinedBehaviorGenerator(context, params, weights)
    elseif behavior_type == "learned"
        behavior_gen = LearnedBehaviorGenerator(driver_network_filepath)
    else
        throw(ArgumentError(
                "invalid behavior type $(behavior_type)"))
    end
    
    models = Dict{Int, DriverModel}()

    # evaluator
    max_num_scenes = Int(ceil((prime_time + sampling_time) / sampling_period))
    rec = SceneRecord(max_num_scenes, sampling_period, max_num_veh)

    # feature_timesteps must be less than the priming steps 
    # (with buffer for features that require 3 timesteps)
    @assert feature_timesteps < Int(floor(prime_time / sampling_period)) - 3

    features = Array{Float64}(feature_dim, feature_timesteps, max_num_veh)
    targets = Array{Float64}(target_dim, max_num_veh)
    agg_targets = Array{Float64}(target_dim, max_num_veh)

    if evaluator_type == "bootstrap"
        if prediction_model_type == "neural_network"
            prediction_model = Network(network_filepath)
        else
            throw(ArgumentError(
                "invalid prediction model type $(prediction_model_type)"))
        end

        eval = BootstrappingMonteCarloEvaluator(ext, num_runs, context, prime_time,
            sampling_time, veh_idx_can_change, rec, features, targets, 
            agg_targets, prediction_model)
    else
        eval = MonteCarloEvaluator(ext, num_runs, context, prime_time, sampling_time,
            veh_idx_can_change, rec, features, targets, agg_targets)
    end

    # dataset
    dataset = Dataset(output_filepath, feature_dim, feature_timesteps, target_dim,
        max_num_samples, chunk_dim = chunk_dim, init_file = false)

    # collector
    col = DatasetCollector(seeds, roadway_gen, scene_gen, behavior_gen, eval,
        dataset, scene, models, roadway, id = col_id)

    return col
end

function get_filepaths(filepath, n)
    dir = dirname(filepath)
    filename = basename(filepath)
    return [string(dir, "/proc_$(i)_$(filename)") for i in 1:n]
end

function build_parallel_dataset_collector(flags)
    num_col = flags["num_proc"]
    output_filepath = flags["output_filepath"]

    filepaths = get_filepaths(output_filepath, num_col)
    cols = [build_dataset_collector(filepaths[i], flags, i) for i in 1:num_col]
    seeds = collect(flags["initial_seed"]:(
        flags["num_scenarios"] + flags["initial_seed"] - 1))
    pcol = ParallelDatasetCollector(cols, seeds, output_filepath)
    return pcol
end
