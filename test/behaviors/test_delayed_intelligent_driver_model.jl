# using Base.Test
# using AutoRisk

function test_delayed_idm()
    # set up a scene with two vehicles in following pattern
    context = IntegratedContinuous(.1, 1)
    num_veh = 2
    
    roadway = gen_straight_roadway(1)

    scene = Scene(num_veh)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    road_pos = 10.
    base_speed = 1.
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(1, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(2, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    rec = SceneRecord(500, .1, num_veh)
    T = 1.

    no_delay_init_scene = copy!(Scene(2), scene)
    with_delay_init_scene = copy!(Scene(2), scene)
    idm_init_scene = copy!(Scene(2), scene)

    # simulate the scene with 0 second time delay idm and compare with normal
    # 0 second delay simulation
    models = Dict{Int, DriverModel}()
    mlon = DelayedIntelligentDriverModel(context.Δt, t_d = 0.0)
    models[1] = Tim2DDriver(context, mlon = mlon)
    models[2] = Tim2DDriver(context, mlon = mlon)
    simulate!(no_delay_init_scene, models, roadway, rec, T)

    # idm simulation
    models = Dict{Int, DriverModel}()
    mlon = IntelligentDriverModel()
    models[1] = Tim2DDriver(context, mlon = mlon)
    models[2] = Tim2DDriver(context, mlon = mlon)
    simulate!(idm_init_scene, models, roadway, rec, T)

    # check vehicle states
    @test no_delay_init_scene == idm_init_scene

    # then run with delay and confirm they are different
    # .2 second delay simulation
    models = Dict{Int, DriverModel}()
    mlon = DelayedIntelligentDriverModel(context.Δt, t_d = 0.2)
    models[1] = Tim2DDriver(context, mlon = mlon)
    models[2] = Tim2DDriver(context, mlon = mlon)
    simulate!(with_delay_init_scene, models, roadway, rec, T)

    @test with_delay_init_scene != idm_init_scene
end

function test_delayed_idm_collision()
    # set up a scene with two vehicles in following pattern
    context = IntegratedContinuous(.1, 1)
    num_veh = 2
    
    roadway = gen_straight_roadway(1)

    scene = Scene(num_veh)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    road_pos = 10.
    
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, 0.)
    veh_state = move_along(veh_state, roadway, road_pos)
    veh_def = VehicleDef(1, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    base_speed = 9.5
    veh_state = VehicleState(Frenet(road_idx, roadway), roadway, base_speed)
    veh_def = VehicleDef(2, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(veh_state, veh_def))

    rec = SceneRecord(500, .1, num_veh)
    T = 1.

    with_delay_init_scene = copy!(Scene(2), scene)
    idm_init_scene = copy!(Scene(2), scene)

    # simulate delayed idm to confirm collision
    models = Dict{Int, DriverModel}()
    mstatic = StaticLongitudinalDriver(0.)
    mlon = DelayedIntelligentDriverModel(context.Δt, t_d = 0.2)
    models[1] = Tim2DDriver(context, mlon = mstatic)
    models[2] = Tim2DDriver(context, mlon = mlon)
    simulate!(with_delay_init_scene, models, roadway, rec, T)
    delayed_idm_in_collision = convert(Float64, get(IS_COLLIDING, rec, roadway, 2))

    # idm simulation that should not have collision
    empty!(rec)
    models = Dict{Int, DriverModel}()
    mstatic = StaticLongitudinalDriver(0.)
    mlon = IntelligentDriverModel()
    models[1] = Tim2DDriver(context, mlon = mstatic)
    models[2] = Tim2DDriver(context, mlon = mlon)
    simulate!(idm_init_scene, models, roadway, rec, T)
    idm_in_collision = convert(Float64, get(IS_COLLIDING, rec, roadway, 2))

    @test delayed_idm_in_collision == 1.
    @test idm_in_collision == 0.
end

@time test_delayed_idm()
@time test_delayed_idm_collision()