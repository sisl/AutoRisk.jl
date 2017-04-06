# using Base.Test
# using AutoRisk

# const NUM_FEATURES = 276
# const NUM_TARGETS = 5
# BASE_TEST_DIR = ".."

function build_debug_base_net_lane_gen()
    num_samples = 1000
    num_vars = 5
    data = ones(Int, num_vars, num_samples) * 2
    data[:,1] = 1
    training_data = DataFrame(
            velocity = data[1,:],
            forevelocity = data[2,:],
            foredistance = data[3,:],
            aggressiveness = data[4,:],
            isattentive = data[5,:]
    )
    base_bn = fit(DiscreteBayesNet, training_data, (
            :isattentive=>:foredistance, 
            :isattentive=>:velocity,
            :aggressiveness=>:foredistance, 
            :aggressiveness=>:velocity,
            :foredistance=>:velocity,
            :forevelocity=>:velocity
        )
    )
    new_values = ones(Int, num_samples)
    new_values[1] = 2
    training_data[:isattentive] = new_values
    prop_bn = fit(DiscreteBayesNet, training_data, (
            :isattentive=>:foredistance, 
            :isattentive=>:velocity,
            :aggressiveness=>:foredistance, 
            :aggressiveness=>:velocity,
            :foredistance=>:velocity,
            :forevelocity=>:velocity
        )
    )
    var_edges = Dict(
        :aggressiveness=>[0.,.5,1.], 
        :foredistance=>[0.,10.,20.],
        :forevelocity=>[0.,5.,10.],
        :velocity=>[0.,5.,10.]
    )
    sampler = UniformAssignmentSampler(var_edges)
    dynamics = Dict(:velocity=>:forevelocity)
    num_veh_per_lane = 2
    min_p = get_passive_behavior_params(err_p_a_to_i = .5)
    max_p = get_aggressive_behavior_params(err_p_a_to_i = .5)
    context = IntegratedContinuous(.1, 1)
    behgen = CorrelatedBehaviorGenerator(context, min_p, max_p)
    gen = BayesNetLaneGenerator(base_bn, prop_bn, sampler, dynamics, num_veh_per_lane, 
        behgen)
    return gen
end

function test_bayes_net_lane_gen_sampling()
    gen = build_debug_base_net_lane_gen()
    roadway = gen_straight_roadway(1)
    scene = Scene(2)
    models = Dict{Int,DriverModel}()
    rand!(gen, roadway, scene, models, 1)

    @test get_weights(gen)[1] < 1.
    @test models[1].is_attentive == false
    @test get_weights(gen)[2] ≈ 1.
    @test models[2].is_attentive == true
end

function test_bayes_net_data_collection()
    gen = build_debug_base_net_lane_gen()
    roadway = gen_straight_roadway(1)
    scene = Scene(2)
    models = Dict{Int,DriverModel}()

    output_filepath = joinpath(BASE_TEST_DIR, "data/test_dataset_collector.h5")
    ext = MultiFeatureExtractor()
    context = IntegratedContinuous(.1, 1)
    num_runs::Int64 = 2
    num_samples = 2
    seeds = collect(1:num_samples)
    prime_time::Float64 = 2.
    sampling_time::Float64 = 3.
    max_num_veh = 2
    target_dim = NUM_TARGETS
    feature_dim = NUM_FEATURES
    veh_idx_can_change::Bool = false
    feature_timesteps = 1
    chunk_dim = 1
    max_num_samples = num_samples * max_num_veh
    max_num_scenes = Int((prime_time + sampling_time) / .1)
    rec::SceneRecord = SceneRecord(max_num_scenes, .1, max_num_veh)
    features::Array{Float64} = Array{Float64}(feature_dim, feature_timesteps,
        max_num_veh)
    targets::Array{Float64} = Array{Float64}(target_dim, max_num_veh)
    agg_targets::Array{Float64} = Array{Float64}(target_dim, max_num_veh)
    rng::MersenneTwister = MersenneTwister(1)
    eval = MonteCarloEvaluator(ext, num_runs, context, prime_time, sampling_time,
        veh_idx_can_change, rec, features, targets, agg_targets, rng)

    # dataset
    dataset = Dataset(output_filepath, feature_dim, feature_timesteps, target_dim,
        max_num_samples, chunk_dim = chunk_dim, use_weights = true)

    # collector
    col = DatasetCollector(seeds, gen, eval, dataset, scene, models, roadway)
    generate_dataset(col)

    file = h5open(output_filepath, "r")
    features = read(file["risk/features"])
    targets = read(file["risk/targets"])
    weights = read(file["risk/weights"])

    @test size(features) == (NUM_FEATURES, 1, 4)
    @test size(targets) == (NUM_TARGETS, 4)
    @test size(weights) == (1, 4)
    @test !any(isnan(features))
    @test !any(isnan(targets))
    @test !any(isnan(weights))

    rm(output_filepath)
end


@time test_bayes_net_lane_gen_sampling()
@time test_bayes_net_data_collection()
