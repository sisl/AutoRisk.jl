# using Base.Test
# using AutoRisk
# using AutoViz
# using Reel
# Reel.set_output_type("gif")

# const NUM_TARGETS = 5

function test_extract_vehicle_frame_targets()
    num_veh = 2
    models = Dict{Int, DriverModel}()
    ext = TargetExtractor()

    # two static drivers not in a collision
    mlon = StaticLaneFollowingDriver(0.)
    models[1] = Tim2DDriver(.1, mlon = mlon)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    roadway = gen_straight_roadway(1, 50.)
    scene = Scene(num_veh)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    road_pos = 10.
    base_speed = 1.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 1))
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 2))
    rec = SceneRecord(500, .1, num_veh)
    T = 1.
    simulate!(LatLonAccel, rec, scene, roadway, models, T)

    targets = Array{Float64}(NUM_TARGETS, 2)
    fill!(targets, 0)
    veh_idx = 1
    t_idx = 1
    targets[:, t_idx] = pull_features!(ext, rec, roadway, veh_idx, 0)
    # extract_vehicle_frame_targets!(rec, roadway, targets, veh_idx, t_idx, 0)
    veh_idx = 2
    t_idx = 2
    targets[:, t_idx] = pull_features!(ext, rec, roadway, veh_idx, 0)
    #extract_vehicle_frame_targets!(rec, roadway, targets, veh_idx, t_idx, 0)

    @test all(abs(targets) .< 1e-8)

    # then in a collision
    scene[2] = Vehicle(VehicleState(Frenet(road_idx, roadway), roadway, 12.), veh_def, 2)
    T = .9
    simulate!(LatLonAccel, rec, scene, roadway, models, T)

    veh_idx = 1
    t_idx = 1
    targets[:, t_idx] = pull_features!(ext, rec, roadway, veh_idx, 0)
    #extract_vehicle_frame_targets!(rec, roadway, targets, veh_idx, t_idx, 0)
    veh_idx = 2
    t_idx = 2
    targets[:, t_idx] = pull_features!(ext, rec, roadway, veh_idx, 0)
    # extract_vehicle_frame_targets!(rec, roadway, targets, veh_idx, t_idx, 0)

    @test targets[2,1] == 1.0
    @test abs(targets[4,1]) < 1e-8
    @test abs(targets[5,1]) < 1e-8

    @test targets[3,2] == 1.0
    @test abs(targets[4,2]) < 1e-8
    @test targets[5,2] == 1.0
end

function test_extract_frame_targets()
    # without changing index
    num_veh = 2
    models = Dict{Int, DriverModel}()
    ext = TargetExtractor()
    mlon = StaticLaneFollowingDriver(0.)
    models[1] = Tim2DDriver(.1, mlon = mlon)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    roadway = gen_straight_roadway(1, 50.)
    scene = Scene(num_veh)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    road_pos = 10.
    base_speed = 1.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 1))
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 2))
    rec = SceneRecord(500, .1, num_veh)
    T = 1.
    simulate!(LatLonAccel, rec, scene, roadway, models, T)

    targets = Array{Float64}(NUM_TARGETS, 2)
    fill!(targets, 0)
    veh_id_to_idx = Dict(1=>1,2=>2)
    veh_idx_can_change = false
    done = Set{Int64}()
    pastframe = 0
    extract_frame_targets!(ext, rec, roadway, targets, veh_id_to_idx, 
        veh_idx_can_change, done, pastframe)

    @test all(abs(targets) .< 1e-8)

    T = .9
    scene[2] = Vehicle(VehicleState(Frenet(road_idx, roadway), roadway, 12.), veh_def, 2)
    simulate!(LatLonAccel, rec, scene, roadway, models, T)

    fill!(targets, 0)
    extract_frame_targets!(ext, rec, roadway, targets, veh_id_to_idx, 
        veh_idx_can_change, done, pastframe)

    @test targets[2,1] == 1.0
    @test abs(targets[4,1]) < 1e-8
    @test abs(targets[5,1]) < 1e-8
    @test targets[3,2] == 1.0
    @test abs(targets[4,2]) < 1e-8
    @test targets[5,2] == 1.0
    @test done == Set([1,2])

    # with changing index
    scene[2] = Vehicle(VehicleState(Frenet(road_idx, roadway), roadway, 13.), veh_def, 2)
    simulate!(LatLonAccel, rec, scene, roadway, models, T)
    done = Set{Int64}()
    veh_idx_can_change = true
    fill!(targets, 0)
    extract_frame_targets!(ext, rec, roadway, targets, veh_id_to_idx, 
        veh_idx_can_change, done, pastframe)

    @test targets[2,1] == 1.0
    @test abs(targets[4,1]) < 1e-8
    @test targets[3,2] == 1.0
    @test abs(targets[4,2]) < 1e-8
    @test done == Set([1,2])
end

function test_extract_targets()

    num_veh = 2
    models = Dict{Int, DriverModel}()
    ext = TargetExtractor()
    mlon = StaticLaneFollowingDriver(0.)
    models[1] = Tim2DDriver(.1, mlon = mlon)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    roadway = gen_straight_roadway(1, 50.)
    scene = Scene(num_veh)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    road_pos = 10.
    base_speed = 1.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 1))
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 2))
    rec = SceneRecord(500, .1, num_veh)
    T = 1.
    simulate!(LatLonAccel, rec, scene, roadway, models, T)
    targets = Array{Float64}(NUM_TARGETS,2)
    fill!(targets, 0)
    veh_id_to_idx = Dict(1=>1,2=>2)
    veh_idx_can_change = true
    extract_targets!(ext, rec, roadway, targets, veh_id_to_idx, veh_idx_can_change)

    @test abs(targets[1,1]) < 1e-8
    @test abs(targets[2,1]) < 1e-8
    @test abs(targets[1,2]) < 1e-8
    @test abs(targets[2,2]) < 1e-8

    fill!(targets, 0)
    scene[2] = Vehicle(VehicleState(Frenet(road_idx, roadway), roadway, 9.5), veh_def, 2)
    models[1] = Tim2DDriver(.1, mlon = StaticLaneFollowingDriver(-5.))
    T = 1.
    simulate!(LatLonAccel, rec, scene, roadway, models, T)
    extract_targets!(ext, rec, roadway, targets, veh_id_to_idx, veh_idx_can_change)

    # frames = Frames(MIME("image/png"), fps=2)
    # frame = render(scene, roadway)
    # push!(frames, frame)
    # write("/Users/wulfebw/Desktop/stuff.gif", frames)
    # println(targets[:,1])
    # println(targets[:,2])

    @test targets[2,1] == 1.0
    @test targets[4,1] == 1.0
    @test targets[3,2] == 1.0
    @test abs(targets[4,2]) < 1e-8
end

function test_pull_features()
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
    mlon = StaticLaneFollowingDriver(1.)
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

    # simulate the scene for 1 second
    rec = SceneRecord(500, .1, num_veh)
    T = 1.

    # simulate here because some features need priming
    simulate!(LatLonAccel, rec, scene, roadway, models, T)
    ext = MultiFeatureExtractor()
    features = Array{Float64}(length(ext), 1, num_veh)
    pull_features!(ext, rec, roadway, models, features)

    @test features[3,1] ≈ 4.
    @test features[4,1] == 5.
    @test features[9,1] ≈ 2.
    @test features[21,1] == 0.
    @test features[22,1] == 0.
    @test features[15,1] ≈ 3.5 / 4.
    @test features[17,1] ≈ 3.5 / 2.

    @test features[3,2] ≈ 2.
    @test features[4,2] == 5.
    @test features[9,2] ≈ 1.
    @test features[15,2] ≈ 3.5 / 2.
    @test features[17,2] ≈ 3.5 / 2.

    @test features[3,3] ≈ 0.
    @test features[4,3] == 5.
    @test features[9,3] ≈ 0.
    @test features[15,3] ≈ 30.0
    @test features[17,3] ≈ 0.0 
end

function test_pull_features_multitimestep()
    num_veh = 2
    timesteps = 2
    roadway = gen_straight_roadway(1, 100.)
    ext = CoreFeatureExtractor()
    rec = SceneRecord(2, .1, num_veh)

    base_speed = 1.
    road_pos = 10.

    scene = Scene(num_veh)
    # 1: first vehicle
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 3., 3.) # check for length
    push!(scene, Vehicle(veh_state, veh_def, 1))
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 5.)
    push!(scene, Vehicle(veh_state, veh_def, 2))
    update!(rec, scene)
    update!(rec, scene) # update second time so features are primed

    # second scene
    scene = Scene(num_veh)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    base_speed = 2.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 6., 6.)
    push!(scene, Vehicle(veh_state, veh_def, 3)) # note the different id
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 5.)
    push!(scene, Vehicle(veh_state, veh_def, 2))
    update!(rec, scene)

    features = Array{Float64}(length(ext), timesteps, num_veh)
    models = Dict{Int, DriverModel}()
    pull_features!(ext, rec, roadway, models, features, timesteps)

    @test features[4,:,1] == [0.,6.]
    @test features[4,:,2] == [5.,5.]
end

@time test_extract_vehicle_frame_targets()
@time test_extract_frame_targets()
@time test_extract_targets()
@time test_pull_features()
@time test_pull_features_multitimestep()