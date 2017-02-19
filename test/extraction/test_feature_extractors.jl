# using Base.Test
# using AutoRisk

function test_car_lidar_feature_extractor()
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

    # simulate here because some features need priming
    T = .6
    rec = SceneRecord(Int(ceil(T/.1)), .1, num_veh)
    simulate!(scene, models, roadway, rec, T)

    carlidar_nbeams = 20
    extract_carlidar_rangerate = true
    ext = CarLidarFeatureExtractor(carlidar_nbeams, 
        extract_carlidar_rangerate = extract_carlidar_rangerate)

    features = zeros(length(ext), num_veh)
    features[:,1] = pull_features!(ext, rec, roadway, 1, models)
    features[:,2] = pull_features!(ext, rec, roadway, 2, models)
    features[:,3] = pull_features!(ext, rec, roadway, 3, models)
    @test features[1,1] ≈ 50.
    @test features[10,1] ≈ 6.72
    @test features[30,1] ≈ -1.6
    @test features[10,2] ≈ 6.72
    @test features[20,2] ≈ 6.72
    @test features[30,2] ≈ -1.6
    @test features[1,3] ≈ 50.
    @test features[20,3] ≈ 6.72
    @test features[30,3] ≈ 0.0
end

function test_normalizing_feature_extractor()
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

    # simulate here because some features need priming
    T = .6
    rec = SceneRecord(Int(ceil(T/.1)), .1, num_veh)
    simulate!(scene, models, roadway, rec, T)

    carlidar_nbeams = 20
    extract_carlidar_rangerate = true
    ext = CarLidarFeatureExtractor(carlidar_nbeams, 
        extract_carlidar_rangerate = extract_carlidar_rangerate)
    ext = NormalizingExtractor(ones(length(ext)), ones(length(ext)), ext)

    features = zeros(length(ext), num_veh)
    features[:, 1] = pull_features!(ext, rec, roadway, 1, models)
    @test features[1,1] ≈ 49.0
    @test features[end,1] ≈ -1.0
end

@time test_car_lidar_feature_extractor()
@time test_normalizing_feature_extractor()