# using Base.Test
# using AutoRisk

function test_delayed_idm()
    # set up a scene with two vehicles in following pattern
    num_veh = 2
    roadway = gen_straight_roadway(1)

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

    no_delay_init_scene = copy!(Scene(2), scene)
    with_delay_init_scene = copy!(Scene(2), scene)
    idm_init_scene = copy!(Scene(2), scene)

    # simulate the scene with 0 second time delay idm and compare with normal
    # 0 second delay simulation
    Δt = .1
    models = Dict{Int, DriverModel}()
    mlon = DelayedIntelligentDriverModel(Δt, t_d = 0.0)
    models[1] = Tim2DDriver(Δt, mlon = mlon)
    models[2] = Tim2DDriver(Δt, mlon = mlon)
    simulate!(LatLonAccel, rec, no_delay_init_scene, roadway, models, T)

    # idm simulation
    models = Dict{Int, DriverModel}()
    mlon = IntelligentDriverModel()
    models[1] = Tim2DDriver(Δt, mlon = mlon)
    models[2] = Tim2DDriver(Δt, mlon = mlon)
    simulate!(LatLonAccel, rec, idm_init_scene, roadway, models, T)

    # check vehicle states
    @test no_delay_init_scene == idm_init_scene

    # then run with delay and confirm they are different
    # .2 second delay simulation
    models = Dict{Int, DriverModel}()
    mlon = DelayedIntelligentDriverModel(Δt, t_d = 0.2)
    models[1] = Tim2DDriver(Δt, mlon = mlon)
    models[2] = Tim2DDriver(Δt, mlon = mlon)
    simulate!(LatLonAccel, rec, with_delay_init_scene, roadway, models, T)

    @test with_delay_init_scene != idm_init_scene
end

function test_delayed_idm_collision()
    # set up a scene with two vehicles in following pattern
    num_veh = 2
    
    roadway = gen_straight_roadway(1)

    scene = Scene(num_veh)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    road_pos = 10.
    
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, 0.)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 1))

    base_speed = 9.5
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def, 2))

    rec = SceneRecord(500, .1, num_veh)
    T = 1.

    with_delay_init_scene = copy!(Scene(2), scene)
    idm_init_scene = copy!(Scene(2), scene)

    # simulate delayed idm to confirm collision
    models = Dict{Int, DriverModel}()
    mstatic = StaticLaneFollowingDriver(0.)
    mlon = DelayedIntelligentDriverModel(.1, t_d = 0.2)
    models[1] = Tim2DDriver(.1, mlon = mstatic)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    simulate!(LatLonAccel, rec, with_delay_init_scene, roadway, models, T)
    delayed_idm_in_collision = convert(Float64, get(IS_COLLIDING, rec, roadway, 2))

    # idm simulation that should not have collision
    empty!(rec)
    models = Dict{Int, DriverModel}()
    mstatic = StaticLaneFollowingDriver(0.)
    mlon = IntelligentDriverModel()
    models[1] = Tim2DDriver(.1, mlon = mstatic)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    simulate!(LatLonAccel, rec, idm_init_scene, roadway, models, T)
    idm_in_collision = convert(Float64, get(IS_COLLIDING, rec, roadway, 2))

    @test delayed_idm_in_collision == 1.
    @test idm_in_collision == 0.
end

@time test_delayed_idm()
@time test_delayed_idm_collision()