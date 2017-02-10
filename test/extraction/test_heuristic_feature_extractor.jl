# using Base.Test
# using AutoRisk

# const NUM_FEATURES = 166
# const NUM_TARGETS = 5

function test_extract_vehicle_features()
    # add three vehicles and specifically check neighbor features
    context = IntegratedContinuous(.1, 1)
    num_veh = 3
    # one lane roadway
    roadway = gen_straight_roadway(1, 100.)
    scene = Scene(num_veh)

    models = Dict{Int, DriverModel}()

    # 1: first vehicle, moving the fastest
    mlon = StaticLongitudinalDriver(2.)
    models[1] = Tim2DDriver(context, mlon = mlon)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    base_speed = 2.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(1, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    # 2: second vehicle, in the middle, moving at intermediate speed
    mlon = StaticLongitudinalDriver(1.0)
    models[2] = Tim2DDriver(context, mlon = mlon)
    base_speed = 1.
    road_pos = 10.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(2, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    # 3: thrid vehicle, in the front, not moving
    mlon = StaticLongitudinalDriver(0.)
    models[3] = Tim2DDriver(context, mlon = mlon)
    base_speed = 0.
    road_pos = 20.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(3, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    # simulate the scene for 1 second
    rec = SceneRecord(500, .1, num_veh)
    T = 1.

    # simulate here because some features need priming
    simulate!(scene, models, roadway, rec, T)
    ext = HeuristicFeatureExtractor()
    features = Array{Float64}(length(ext), num_veh)

    pull_features!(ext, features, rec, roadway, 1, models)

    @test features[3,1] ≈ 4.
    @test features[4,1] == 5.
    @test features[6,1] ≈ 2.
    @test features[21,1] == 0.
    @test features[22,1] == 0.
    @test features[23,1] ≈ 3.5 / 4.
    @test features[25,1] ≈ 3.5 / 2.

    pull_features!(ext, features, rec, roadway, 2, models)

    @test features[3,2] ≈ 2.
    @test features[4,2] == 5.
    @test features[6,2] ≈ 1.
    @test features[23,2] ≈ 3.5 / 2.
    @test features[25,2] ≈ 3.5 / 2.

    pull_features!(ext, features, rec, roadway, 3, models)

    @test features[3,3] ≈ 0.
    @test features[4,3] == 5.
    @test features[6,3] ≈ 0.
    @test features[23,3] ≈ 30.
    @test features[25,3] ≈ 0. 
end

@time test_extract_vehicle_features()