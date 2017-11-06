# using Base.Test
# using AutoRisk

# const NUM_FEATURES = 268
# const NUM_TARGETS = 5

function test_bootstrapping_monte_carlo_evaluator_debug()
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

    # 3: thrid vehicle, in the front, accelerating backward
    mlon = StaticLaneFollowingDriver(-4.5)
    models[3] = Tim2DDriver(.1, mlon = mlon)
    base_speed = 0.
    road_pos = 200.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 3))

    num_runs::Int64 = 10
    prime_time::Float64 = 1.
    sampling_time::Float64 = 1.
    veh_idx_can_change::Bool = false

    rec::SceneRecord = SceneRecord(500, .1, num_veh)
    features::Array{Float64} = Array{Float64}(NUM_FEATURES, 1,num_veh)
    targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)
    agg_targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)

    rng::MersenneTwister = MersenneTwister(1)
    prediction_model = Network()
    prediction_model.means = zeros(Float64, (1, NUM_FEATURES))
    prediction_model.stds = ones(Float64, (1, NUM_FEATURES))
    push!(prediction_model.weights, ones(Float64, (NUM_FEATURES, NUM_TARGETS)))
    push!(prediction_model.biases, zeros(Float64, (1, NUM_TARGETS)))

    ext = MultiFeatureExtractor()
    target_ext = TargetExtractor()
    eval = BootstrappingMonteCarloEvaluator(ext, target_ext, num_runs, prime_time,
        sampling_time, veh_idx_can_change, rec, features, targets, agg_targets,
        prediction_model, rng)

    evaluate!(eval, scene, models, roadway, 1)

    # note that the bootstrapping model will output ones for everything
    # which means that everything that's already collided will have normal 
    # values (vehicles 1 and 2), and everything that hasn't collided will 
    # have all ones (vehicle 3)
    @test eval.agg_targets[1:NUM_TARGETS, 1] == [0.0, 0.0, 1.0, 0.0, 1.0]
    @test eval.agg_targets[1:NUM_TARGETS, 2] == [0.0, 1.0, 0.0, 0.0, 0.0]
    @test eval.agg_targets[1:NUM_TARGETS, 3] == [1.0, 1.0, 1.0, 1.0, 1.0]
end

# @time test_bootstrapping_monte_carlo_evaluator_debug()