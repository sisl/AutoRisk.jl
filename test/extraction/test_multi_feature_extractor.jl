# using Base.Test
# using AutoRisk

function test_multi_feature_extractor_heuristic()
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
    ext = MultiFeatureExtractor()
    features = Array{Float64}(length(ext), num_veh)

    features[:, 1] = pull_features!(ext, rec, roadway, 1, models)

    @test features[3,1] ≈ 4.
    @test features[4,1] == 5.

    @test features[9,1] ≈ 2.
    @test features[15,1] ≈ 3.5 / 4.
    @test features[17,1] ≈ 3.5 / 2.
    @test features[19,1] == 0.

    features[:, 2] = pull_features!(ext, rec, roadway, 2, models)

    @test features[3,2] ≈ 2.
    @test features[4,2] == 5.
    @test features[9,2] ≈ 1.
    @test features[15,2] ≈ 3.5 / 2.
    @test features[17,2] ≈ 3.5 / 2.

    features[:, 3] = pull_features!(ext, rec, roadway, 3, models)

    @test features[3,3] ≈ 0.
    @test features[4,3] == 5.
    @test features[9,3] ≈ 0.
    @test features[15,3] ≈ 30.
    @test features[17,3] ≈ 0. 
end

function test_multi_feature_extractor()
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
    veh_def = VehicleDef(1, AgentClass.CAR, 4., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    # 2: second vehicle, in the middle, moving at intermediate speed
    mlon = StaticLongitudinalDriver(1.0)
    models[2] = Tim2DDriver(context, mlon = mlon)
    base_speed = 1.
    road_pos = 10.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(2, AgentClass.CAR, 4.5, 2.)
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

    subexts = [
        CoreFeatureExtractor(),
        TemporalFeatureExtractor(),
        WellBehavedFeatureExtractor(),
        NeighborFeatureExtractor(),
        CarLidarFeatureExtractor(),
        RoadLidarFeatureExtractor()
    ]
    ext = MultiFeatureExtractor(subexts)
    @test length(ext) == 140
    features = Array{Float64}(length(ext), num_veh)

    features[:,1] = pull_features!(ext, rec, roadway, 1, models)
    features[:,2] = pull_features!(ext, rec, roadway, 2, models)
    features[:,3] = pull_features!(ext, rec, roadway, 3, models)

    @test features[4,1] ≈ 4.
    @test features[4,2] ≈ 4.5
    @test features[4,3] ≈ 5.

    @test features[9,1] ≈ 2.
    @test features[9,2] ≈ 1.
    @test features[9,3] ≈ 0.
end

function test_feature_names()
    ext = MultiFeatureExtractor()
    fs = feature_names(ext)
    @test length(fs) == length(ext)
    @test fs[3] == "velocity"
    print(fs)
end

@time test_multi_feature_extractor_heuristic()
@time test_multi_feature_extractor()
@time test_feature_names()
