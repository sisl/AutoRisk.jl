# using Base.Test
# using AutoRisk

function get_feature_debug_scenario()
    # add three vehicles and specifically check neighbor features
    num_veh = 3
    # one lane roadway
    roadway = gen_straight_roadway(1, 100.)
    scene = Scene(num_veh)

    models = Dict{Int, DriverModel}()

    # 1: first vehicle, moving the fastest
    mlon = StaticLaneFollowingDriver(2.)
    models[1] = Tim2DDriver(.1, mlon = mlon)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    base_speed = 2.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 1))

    # 2: second vehicle, in the middle, moving at intermediate speed
    mlon = StaticLaneFollowingDriver(1.0)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    base_speed = 1.
    road_pos = 10.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 2))

    # 3: thrid vehicle, in the front, not moving
    mlon = StaticLaneFollowingDriver(0.)
    models[3] = Tim2DDriver(.1, mlon = mlon)
    base_speed = 0.
    road_pos = 20.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 3))
    return num_veh, scene, roadway, models
end

function test_car_lidar_feature_extractor()
    num_veh, scene, roadway, models = get_feature_debug_scenario()

    # simulate here because some features need priming
    T = .6
    nticks = Int(ceil(T/.1))
    rec = SceneRecord(nticks, .1, num_veh)
    simulate!(LatLonAccel, rec, scene, roadway, models, T)

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
    
    num_veh, scene, roadway, models = get_feature_debug_scenario()

    # simulate here because some features need priming
    T = .6
    nticks = Int(ceil(T/.1))
    rec = SceneRecord(nticks, .1, num_veh)
    simulate!(LatLonAccel, rec, scene, roadway, models, T)

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

function test_neighbor_and_temporal_feature_extractors()
    num_veh, scene, roadway, models = get_feature_debug_scenario()

    # set the first and third modles to accelerate
    models[1] = Tim2DDriver(.1, mlon=IntelligentDriverModel(σ=.1))
    models[2] = Tim2DDriver(.1, mlon=StaticLaneFollowingDriver(-.1))
    models[3] = Tim2DDriver(.1, mlon=IntelligentDriverModel(σ=.1))

    # simulate here because some features need priming
    T = .5
    feature_timesteps = 2
    nticks = Int(ceil(T/.1)) * 2
    rec = SceneRecord(nticks, .1, num_veh)
    simulate!(LatLonAccel, rec, scene, roadway, models, T)

    subexts = AbstractFeatureExtractor[
        NeighborFeatureExtractor(),
        TemporalFeatureExtractor()
    ]
    ext = MultiFeatureExtractor(subexts)

    features = zeros(length(ext), feature_timesteps, num_veh)
    features[:,1,1] = pull_features!(ext, rec, roadway, 1, models)
    features[:,1,2] = pull_features!(ext, rec, roadway, 2, models)
    features[:,1,3] = pull_features!(ext, rec, roadway, 3, models)
    features[:,2,1] = pull_features!(ext, rec, roadway, 1, models, -2)
    features[:,2,2] = pull_features!(ext, rec, roadway, 2, models, -2)
    features[:,2,3] = pull_features!(ext, rec, roadway, 3, models, -2)

    fns = feature_names(ext)
    # neighbor
    fns_to_test = [["$(v)_m_dist", "$(v)_m_vel", "$(v)_m_accel"] 
        for v in ["fore", "rear"]]
    fns_to_test = [fns_to_test[1] ; fns_to_test[2]]
    for fn in fns_to_test
        fidx = find(fns .== fn)
        try
            @test features[fidx,1,2] != features[fidx,2,2]
        catch e
            println("neighbor feature extractor failed")
            println("feature $(fn) stayed the same across timesteps")
            throw(e)
        end
    end

    # temporal
    fidx = find(fns .== "timegap")
    @test features[fidx,1,1] != features[fidx,2,1]
    @test features[fidx,1,2] != features[fidx,2,2]
    # fidx = find(fns .== "time_to_collision")
    # println(features[fidx,:,1])
    # println(features[fidx,:,2])
    # println(features[fidx,:,3])
    # @test features[fidx,1,1] != features[fidx,2,1] 
end

function test_feature_step_size_larger_than_1()
    num_veh, scene, roadway, models = get_feature_debug_scenario()

    # set the first and third modles to accelerate
    models[1] = Tim2DDriver(.1, mlon=StaticLaneFollowingDriver(1.))
    init_v_id_1 = scene[1].state.v
    models[2] = Tim2DDriver(.1, mlon=StaticLaneFollowingDriver(2.))
    init_v_id_2 = scene[2].state.v
    models[3] = Tim2DDriver(.1, mlon=StaticLaneFollowingDriver(3.))
    init_v_id_3 = scene[3].state.v

    # simulate here because some features need priming
    T = .4
    nticks = Int(ceil(T/.1))
    rec = SceneRecord(nticks + 1, .1, num_veh)
    simulate!(LatLonAccel, rec, scene, roadway, models, T, update_first_scene = true)

    ext = CoreFeatureExtractor()
    # frames: [.4 .3 .2 .1 .0]
    # following should extract features for [.4 .2 .0]
    # for veh 1, this should give init_v_id_1 + [0, .2, .4]
    # for veh 2, init_v_id_2 + [0, .4, .8]
    # for veh 3, init_v_id_3 + [0, .6, 1.2]
    timesteps = 3
    step_size = 2
    features = zeros(length(ext), timesteps, num_veh)
    pull_features!(ext, rec, roadway, models, features, timesteps, step_size = 2)
    vel_idx = find(feature_names(ext) .== "velocity")

    @test all(features[vel_idx,:,1] .- (init_v_id_1 + [0 .2 .4]) .< 0.0001)
    @test all(features[vel_idx,:,2] .- (init_v_id_2 + [0 .4 .8]) .< 0.0001)
    @test all(features[vel_idx,:,3] .- (init_v_id_3 + [0 .6 1.2]) .< 0.0001)
end

function test_feature_info()
    ext = CoreFeatureExtractor()
    info = feature_info(ext)

    @test in("length", keys(info))
    @test info["length"]["high"] == 30.
    @test info["length"]["low"] == 2.

    ext = MultiFeatureExtractor()
    info = feature_info(ext)

    @test in("length", keys(info))
    @test info["length"]["high"] == 30.
    @test info["length"]["low"] == 2.

    for (k,v) in info
        @test typeof(v["high"]) == Float64
        @test typeof(v["low"]) == Float64
    end
end

@time test_car_lidar_feature_extractor()
@time test_normalizing_feature_extractor()
@time test_neighbor_and_temporal_feature_extractors()
@time test_feature_step_size_larger_than_1()
@time test_feature_info()