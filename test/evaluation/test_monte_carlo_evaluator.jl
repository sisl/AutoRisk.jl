# using Base.Test
# using AutoRisk

# # using AutoViz
# # using Reel
# # Reel.set_output_type("gif")

# NUM_FEATURES = 324
# NUM_TARGETS = 5

function test_monte_carlo_evaluator_debug()
    # add three vehicles and specifically check neighbor features
    num_veh = 3
    # one lane roadway
    roadway = gen_straight_roadway(1, 500.)
    scene = Scene(num_veh)

    models = Dict{Int, DriverModel}()

    # 1: first vehicle, moving the fastest
    mlon = StaticLaneFollowingDriver(5.)
    models[1] = Tim2DDriver(.1, mlon = mlon)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    base_speed = 2.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 1))

    # 2: second vehicle, in the middle, moving at intermediate speed
    mlon = StaticLaneFollowingDriver(1.)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    base_speed = 1.
    road_pos = 10.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 2))

    # 3: thrid vehicle, in the front, moving backward
    mlon = StaticLaneFollowingDriver(-4.5)
    models[3] = Tim2DDriver(.1, mlon = mlon)
    base_speed = 0.
    road_pos = 200.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 3))

    # frames = Frames(MIME("image/png"), fps=2)
    # frame = render(scene, roadway)
    # push!(frames, frame)
    # write("/Users/wulfebw/Desktop/stuff.gif", frames)

    num_runs::Int64 = 10
    prime_time::Float64 = 1.
    sampling_time::Float64 = 1.
    veh_idx_can_change::Bool = false

    rec::SceneRecord = SceneRecord(500, .1, num_veh)
    features::Array{Float64} = Array{Float64}(NUM_FEATURES, 1,num_veh)
    targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)
    agg_targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)

    rng::MersenneTwister = MersenneTwister(1)
    ext = MultiFeatureExtractor()
    target_ext = TargetExtractor()
    eval = MonteCarloEvaluator(ext, target_ext, num_runs, prime_time, sampling_time,
        veh_idx_can_change, rec, features, targets, agg_targets, rng)

    evaluate!(eval, scene, models, roadway, 1)

    # first two collisions in each, last decel in each
    @test eval.agg_targets[1:NUM_TARGETS, 1] == [0.0, 0.0, 1.0, 0.0, 1.0]
    @test eval.agg_targets[1:NUM_TARGETS, 2] == [0.0, 1.0, 0.0, 0.0, 0.0]
    @test eval.agg_targets[1:NUM_TARGETS, 3] == [0.0, 0.0, 0.0, 1.0, 0.0]
end

function test_monte_carlo_evaluator()
    num_veh = 2
    # one lane roadway
    roadway = gen_straight_roadway(1, 500.)
    scene = Scene(num_veh)

    models = Dict{Int, DriverModel}()
    k_spd = 1.
    politeness = 3.

    # 1: first vehicle, moving the fastest
    mlane = MOBIL(.1, politeness = politeness)
    mlon = IntelligentDriverModel(k_spd = k_spd)
    models[1] = Tim2DDriver(.1, mlane = mlane, mlon = mlon)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    base_speed = 10.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 1))

    # 2: second vehicle, in the middle, moving at intermediate speed
    mlane = MOBIL(.1, politeness = politeness)
    mlon = IntelligentDriverModel(k_spd = k_spd)
    models[2] = Tim2DDriver(.1, mlane = mlane, mlon = mlon)
    base_speed = 0.
    road_pos = 8.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 2))

    num_runs::Int64 = 10
    prime_time::Float64 = .2
    sampling_time::Float64 = 3.
    veh_idx_can_change::Bool = false

    rec::SceneRecord = SceneRecord(500, .1, num_veh)
    features::Array{Float64} = Array{Float64}(NUM_FEATURES, 1, num_veh)
    targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)
    agg_targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)

    rng::MersenneTwister = MersenneTwister(1)

    ext = MultiFeatureExtractor()
    target_ext = TargetExtractor()
    eval = MonteCarloEvaluator(ext, target_ext, num_runs, prime_time, sampling_time,
        veh_idx_can_change, rec, features, targets, agg_targets, rng)

    evaluate!(eval, scene, models, roadway, 1)

    feature_names_list = feature_names(ext)

    @test eval.agg_targets[1:NUM_TARGETS, 1] == [0.0, 0.0, 1.0, 1.0, 1.0]
    @test eval.agg_targets[1:NUM_TARGETS, 2] == [0.0, 1.0, 0.0, 0.0, 0.0]

    @test eval.features[15, 1] ≈ 0.151219512195122
    @test eval.features[15, 2] ≈ 30.

    @test eval.features[17, 1] ≈ 1. / 6.12903225806451
    @test eval.features[17, 2] ≈ 30.0

    k_spd_idx = find(feature_names_list .== "beh_lon_k_spd")[1]

    @test eval.features[k_spd_idx, 1] == k_spd
    @test eval.features[k_spd_idx, 2] == k_spd

    politeness_idx = find(feature_names_list .== "beh_lane_politeness")[1]

    @test eval.features[politeness_idx, 1] == politeness
    @test eval.features[politeness_idx, 2] == politeness

end

function test_multi_timestep_monte_carlo_evaluator()
    num_veh = 2

    # one lane roadway
    roadway = gen_straight_roadway(1, 500.)
    scene = Scene(num_veh)
    models = Dict{Int, DriverModel}()

    # 1: first vehicle, accelerating the fastest
    mlon = StaticLaneFollowingDriver(2.)
    models[1] = Tim2DDriver(.1, mlon = mlon)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    base_speed = 0.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 1))

    # 2: second vehicle, in the middle, accelerating at intermediate speed
    mlon = StaticLaneFollowingDriver(1.)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    base_speed = 0.
    road_pos = 10.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 2))

    num_runs::Int64 = 2
    prime_time::Float64 = .5
    sampling_time::Float64 = 1.
    veh_idx_can_change::Bool = false
    feature_timesteps = 5

    rec::SceneRecord = SceneRecord(500, .1, num_veh)
    rng::MersenneTwister = MersenneTwister(1)

    ext = MultiFeatureExtractor()
    target_ext = TargetExtractor()

    features::Array{Float64} = Array{Float64}(length(ext), feature_timesteps,
        num_veh)
    targets::Array{Float64} = Array{Float64}(length(target_ext), num_veh)
    agg_targets::Array{Float64} = Array{Float64}(length(target_ext), num_veh)

    eval = MonteCarloEvaluator(ext, target_ext, num_runs, prime_time, sampling_time,
        veh_idx_can_change, rec, features, targets, agg_targets, rng)
    original_scene = copy!(Scene(num_veh), scene)
    evaluate!(eval, scene, models, roadway, 1)

    @test size(eval.features) == (length(ext), feature_timesteps, 2)
    @test size(eval.targets) == (length(target_ext), 2)

    # check velocity over time
    @test eval.features[3,1,1] ≈ .2
    @test eval.features[3,2,1] ≈ .4
    @test eval.features[3,3,1] ≈ .6
    @test eval.features[3,4,1] ≈ .8
    @test eval.features[3,5,1] ≈ 1.
    @test eval.features[3,1,2] ≈ .1
    @test eval.features[3,2,2] ≈ .2
    @test eval.features[3,3,2] ≈ .3
    @test eval.features[3,4,2] ≈ .4
    @test eval.features[3,5,2] ≈ .5

    # check accel over time
    @test eval.features[9,1,1] ≈ 2.
    @test eval.features[9,2,1] ≈ 2.
    @test eval.features[9,3,1] ≈ 2.
    @test eval.features[9,1,2] ≈ 1.
    @test eval.features[9,2,2] ≈ 1.
    @test eval.features[9,3,2] ≈ 1.

    # now check for step_size case
    step_size = 2
    feature_timesteps = 2
    features = Array{Float64}(length(ext), feature_timesteps,
        num_veh)
    eval = MonteCarloEvaluator(ext, target_ext, num_runs, prime_time, sampling_time,
        veh_idx_can_change, rec, features, targets, agg_targets, rng, 
        feature_step_size = step_size)
    evaluate!(eval, original_scene, models, roadway, 1)

    @test size(eval.features) == (length(ext), feature_timesteps, 2)
    @test size(eval.targets) == (length(target_ext), 2)

    # check velocity over time
    @test eval.features[3,1,1] ≈ .6
    @test eval.features[3,2,1] ≈ 1.
    @test eval.features[3,1,2] ≈ .3
    @test eval.features[3,2,2] ≈ .5

    # check accel over time
    @test eval.features[9,1,1] ≈ 2.
    @test eval.features[9,2,1] ≈ 2.
    @test eval.features[9,1,2] ≈ 1.
    @test eval.features[9,2,2] ≈ 1.

end

@time test_monte_carlo_evaluator_debug()
@time test_monte_carlo_evaluator()
@time test_multi_timestep_monte_carlo_evaluator()