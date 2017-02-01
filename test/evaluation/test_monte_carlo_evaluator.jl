# using Base.Test
# using AutomotiveDrivingModels

# push!(LOAD_PATH, "../../src")
# include("../../src/evaluation/simulation.jl")
# include("../../src/utils/automotive.jl")
# include("../../src/evaluation/dataset_extraction.jl")
# include("../../src/evaluation/monte_carlo_evaluator.jl")

function test_monte_carlo_evaluator_debug()
    # add three vehicles and specifically check neighbor features
    context = IntegratedContinuous(.1, 1)
    num_veh = 3
    # one lane roadway
    roadway = gen_straight_roadway(1, 500.)
    scene = Scene(num_veh)

    models = Dict{Int, DriverModel}()

    # 1: first vehicle, moving the fastest
    mlon = StaticLongitudinalDriver(5.)
    models[1] = Tim2DDriver(context, mlon = mlon)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    base_speed = 2.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(1, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    # 2: second vehicle, in the middle, moving at intermediate speed
    mlon = StaticLongitudinalDriver(1.)
    models[2] = Tim2DDriver(context, mlon = mlon)
    base_speed = 1.
    road_pos = 10.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(2, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    # 3: thrid vehicle, in the front, not moving
    mlon = StaticLongitudinalDriver(-4.5)
    models[3] = Tim2DDriver(context, mlon = mlon)
    base_speed = 0.
    road_pos = 200.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(3, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    num_runs::Int64 = 10
    prime_time::Float64 = 1.
    sampling_time::Float64 = 1.
    veh_idx_can_change::Bool = false

    rec::SceneRecord = SceneRecord(500, .1, num_veh)
    features::Array{Float64} = Array{Float64}(NUM_FEATURES, num_veh)
    targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)
    agg_targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)

    rng::MersenneTwister = MersenneTwister(1)

    eval = MonteCarloEvaluator(num_runs, context, prime_time, sampling_time,
        veh_idx_can_change, rec, features, targets, agg_targets, rng)

    evaluate!(eval, scene, models, roadway, 1)

    # first two collisions in each, last decel in each
    @test eval.agg_targets[1:NUM_TARGETS, 1] == [0.0, 0.0, 1.0, 0.0, 1.0]
    @test eval.agg_targets[1:NUM_TARGETS, 2] == [0.0, 1.0, 0.0, 0.0, 0.0]
    @test eval.agg_targets[1:NUM_TARGETS, 3] == [0.0, 0.0, 0.0, 1.0, 0.0]
end

function test_monte_carlo_evaluator()
    context = IntegratedContinuous(.1, 1)
    num_veh = 2
    # one lane roadway
    roadway = gen_straight_roadway(1, 500.)
    scene = Scene(num_veh)

    models = Dict{Int, DriverModel}()
    k_spd = 1.
    politeness = 3.

    # 1: first vehicle, moving the fastest
    mlane = MOBIL(context, politeness = politeness)
    mlon = IntelligentDriverModel(k_spd = k_spd)
    models[1] = Tim2DDriver(context, mlane = mlane, mlon = mlon)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    base_speed = 10.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(1, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    # 2: second vehicle, in the middle, moving at intermediate speed
    mlane = MOBIL(context, politeness = politeness)
    mlon = IntelligentDriverModel(k_spd = k_spd)
    models[2] = Tim2DDriver(context, mlane = mlane, mlon = mlon)
    base_speed = 0.
    road_pos = 8.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(2, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    num_runs::Int64 = 10
    prime_time::Float64 = .2
    sampling_time::Float64 = 3.
    veh_idx_can_change::Bool = false

    rec::SceneRecord = SceneRecord(500, .1, num_veh)
    features::Array{Float64} = Array{Float64}(NUM_FEATURES, num_veh)
    targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)
    agg_targets::Array{Float64} = Array{Float64}(NUM_TARGETS, num_veh)

    rng::MersenneTwister = MersenneTwister(1)

    eval = MonteCarloEvaluator(num_runs, context, prime_time, sampling_time,
        veh_idx_can_change, rec, features, targets, agg_targets, rng)

    evaluate!(eval, scene, models, roadway, 1)

    @test eval.agg_targets[1:NUM_TARGETS, 1] == [0.0, 0.0, 1.0, 1.0, 1.0]
    @test eval.agg_targets[1:NUM_TARGETS, 2] == [0.0, 1.0, 0.0, 0.0, 0.0]

    @test eval.features[23, 1] ≈ 0.151219512195122
    @test eval.features[23, 2] ≈ 30.

    @test eval.features[25, 1] ≈ 1. / 6.12903225806451
    @test eval.features[25, 2] ≈ 0.0

    @test eval.features[62, 1] == k_spd
    @test eval.features[62, 2] == k_spd

    @test eval.features[72, 1] == politeness
    @test eval.features[72, 2] == politeness

end

@time test_monte_carlo_evaluator_debug()
@time test_monte_carlo_evaluator()