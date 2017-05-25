# using Base.Test
# using AutoRisk

# const NUM_FEATURES = 268
# const NUM_TARGETS = 5
# BASE_TEST_DIR = ".."

function build_debug_base_net_lane_gen()
    num_samples = 1000
    num_vars = 7
    data = ones(Int, num_vars, num_samples) * 2
    data[:,1] = 1
    training_data = DataFrame(
            relvelocity = data[1,:],
            forevelocity = data[2,:],
            foredistance = data[3,:],
            aggressiveness = data[4,:],
            isattentive = data[5,:],
            vehlength = data[6,:],
            vehwidth = data[7,:]
    )
    base_bn = fit(DiscreteBayesNet, training_data, (
            :isattentive=>:foredistance, 
            :isattentive=>:relvelocity,
            :aggressiveness=>:foredistance, 
            :aggressiveness=>:relvelocity,
            :foredistance=>:relvelocity,
            :forevelocity=>:relvelocity,
            :vehlength=>:vehwidth
        )
    )
    new_values = ones(Int, num_samples)
    new_values[1] = 2
    training_data[:isattentive] = new_values
    prop_bn = fit(DiscreteBayesNet, training_data, (
            :isattentive=>:foredistance, 
            :isattentive=>:relvelocity,
            :aggressiveness=>:foredistance, 
            :aggressiveness=>:relvelocity,
            :foredistance=>:relvelocity,
            :forevelocity=>:relvelocity,
            :vehlength=>:vehwidth
        )
    )
    discs = Dict{Symbol, LinCatDiscretizer}(
        :aggressiveness=>LinearDiscretizer([0.,.5,1.]), 
        :foredistance=>LinearDiscretizer([0.,10.,20.]),
        :forevelocity=>LinearDiscretizer([0.,5.,10.]),
        :relvelocity=>LinearDiscretizer([0.,5.,10.]),
        :isattentive=>CategoricalDiscretizer([1,2]),
        :aggressiveness=>LinearDiscretizer([0.,.5,1.]),
        :vehlength=>LinearDiscretizer([0.,.5,3.]),
        :vehwidth=>LinearDiscretizer([0.,.5,3.])
    )
    
    sampler = AssignmentSampler(discs)
    num_veh_per_lane = 2
    min_p = get_passive_behavior_params(err_p_a_to_i = .5)
    max_p = get_aggressive_behavior_params(err_p_a_to_i = .5)
    behgen = CorrelatedBehaviorGenerator(min_p, max_p)
    gen = BayesNetLaneGenerator(base_bn, sampler, prop_bn, sampler, 
        num_veh_per_lane, behgen)
    return gen
end

function test_bayes_net_lane_gen_sampling()
    gen = build_debug_base_net_lane_gen()
    roadway = gen_straight_roadway(1)
    scene = Scene(2)
    models = Dict{Int,DriverModel}()
    rand!(gen, roadway, scene, models, 1)

    # proposal car is the first one
    @test get_weights(gen)[1] < 1.
    @test models[1].is_attentive == false
    @test get_weights(gen)[2] ≈ 1.
    @test models[2].is_attentive == true
end

function run_bayes_net_collection()
    gen = build_debug_base_net_lane_gen()
    roadway = gen_straight_roadway(1)
    scene = Scene(2)
    models = Dict{Int,DriverModel}()

    ext = MultiFeatureExtractor()
    num_runs = 2
    num_samples = 2
    seeds = collect(1:num_samples)
    prime_time = .5
    sampling_time = .5
    max_num_veh = 2
    target_dim = NUM_TARGETS
    feature_dim = NUM_FEATURES
    veh_idx_can_change = false
    feature_timesteps = 1
    chunk_dim = 1
    max_num_samples = num_samples * max_num_veh
    max_num_scenes = Int((prime_time + sampling_time) / .1)
    rec = SceneRecord(max_num_scenes, .1, max_num_veh)
    features = Array{Float64}(feature_dim, feature_timesteps, max_num_veh)
    targets = Array{Float64}(target_dim, max_num_veh)
    agg_targets = Array{Float64}(target_dim, max_num_veh)
    rng = MersenneTwister(1)
    eval = MonteCarloEvaluator(ext, num_runs, prime_time, sampling_time,
        veh_idx_can_change, rec, features, targets, agg_targets, rng)

    # dataset
    output_filepath = joinpath(BASE_TEST_DIR, "data/test_dataset_collector.h5")
    dataset = Dataset(output_filepath, feature_dim, feature_timesteps, target_dim,
        max_num_samples, chunk_dim = chunk_dim, use_weights = true)

    # collector
    col = DatasetCollector(seeds, gen, eval, dataset, scene, models, roadway)
    generate_dataset(col)

    file = h5open(output_filepath, "r")
    features = read(file["risk/features"])
    targets = read(file["risk/targets"])
    weights = read(file["risk/weights"])
    close(file)
    rm(output_filepath)

    return features, targets, weights
end

function test_bayes_net_data_collection()
    srand(1)
    features_1, targets_1, weights_1 = run_bayes_net_collection()

    @test size(features_1) == (NUM_FEATURES, 1, 4)
    @test size(targets_1) == (NUM_TARGETS, 4)
    @test size(weights_1) == (1, 4)
    @test !any(isnan(features_1))
    @test !any(isnan(targets_1))
    @test !any(isnan(weights_1))

    # check deterministic
    srand(1)
    features_2, targets_2, weights_2 = run_bayes_net_collection()

    @test features_1 ≈ features_2
    @test targets_1 ≈ targets_2
    @test weights_1 ≈ weights_2
end

function build_simple_realistic_base_net_lane_gen(;
        num_veh_per_lane = 2
    )
    num_samples = 1000
    num_vars = 7
    # each variable equally split between bins
    data = ones(Int, num_vars, num_samples)
    data[:,Int(ceil(end/2)):end] = 2

    training_data = DataFrame(
            relvelocity = data[1,:],
            forevelocity = data[2,:],
            foredistance = data[3,:],
            aggressiveness = data[4,:],
            isattentive = data[5,:],
            vehlength = data[6,:],
            vehwidth = data[7,:]
    )
    base_bn = fit(DiscreteBayesNet, training_data, (
            :isattentive=>:foredistance, 
            :isattentive=>:relvelocity,
            :aggressiveness=>:foredistance, 
            :aggressiveness=>:relvelocity,
            :foredistance=>:relvelocity,
            :forevelocity=>:relvelocity,
            :vehlength=>:vehwidth
        )
    )
    prop_bn = base_bn
    discs = Dict{Symbol, LinCatDiscretizer}(
        :aggressiveness=>LinearDiscretizer([0.,.5,1.]), 
        :foredistance=>LinearDiscretizer([5.,5.1, 5.2]),
        :forevelocity=>LinearDiscretizer([1.,5.,10.]),
        :relvelocity=>LinearDiscretizer([-1.,0.,1.]),
        :isattentive=>CategoricalDiscretizer([1,2]),
        :aggressiveness=>LinearDiscretizer([0.,.5,1.]),
        :vehlength=>LinearDiscretizer([0.,1.,2.]),
        :vehwidth=>LinearDiscretizer([0.,1.,2.])
    )
    
    sampler = AssignmentSampler(discs)
    min_p = get_passive_behavior_params()
    max_p = get_aggressive_behavior_params()
    behgen = CorrelatedBehaviorGenerator(min_p, max_p)
    gen = BayesNetLaneGenerator(base_bn, sampler, prop_bn, sampler, 
        num_veh_per_lane, behgen)
    return gen
end

function test_scene_features_align_with_bounds()
    num_veh_per_lane = 3
    gen = build_simple_realistic_base_net_lane_gen(
        num_veh_per_lane = num_veh_per_lane
    )
    roadway = gen_straight_roadway(1)
    scene = Scene(num_veh_per_lane)
    models = Dict{Int,DriverModel}()
    rand!(gen, roadway, scene, models, 2)
    ext = NeighborFeatureExtractor()
    rec = SceneRecord(num_veh_per_lane, .1)
    update!(rec, scene)
    features = Array{Float64}(length(ext), num_veh_per_lane)
    features[:, 1] = pull_features!(ext, rec, roadway, 1, models)
    features[:, 2] = pull_features!(ext, rec, roadway, 2, models)
    features[:, 3] = pull_features!(ext, rec, roadway, 3, models)

    @test 5.1 <= features[5, 2] <= 5.2
    @test 1. <= features[6, 2] <= 10.
    @test 5.0 <= features[5, 3] <= 5.1
    @test 1. <= features[6, 3] <= 10.
end

@time test_bayes_net_lane_gen_sampling()
@time test_bayes_net_data_collection()
@time test_scene_features_align_with_bounds()