# using Base.Test
# using AutoRisk

function test_delayed_driver_observe()
    context = IntegratedContinuous(.1, 1)
    roadway = gen_straight_roadway(1)
    num_veh = 2
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

    # test without reaction time
    model = DelayedDriver(Tim2DDriver(context), reaction_time = 0.0)
    observe!(model, scene, roadway, 1)
    @test model.rec.nscenes == 1
    @test length(model.rec.scenes) == 1
    @test model.driver.mlane.rec.nscenes == 1

    # test with reaction time
    model = DelayedDriver(Tim2DDriver(context), reaction_time = 0.1)
    observe!(model, scene, roadway, 1)
    @test model.rec.nscenes == 1
    @test length(model.rec.scenes) == 2
    @test model.driver.mlane.rec.nscenes == 0
    observe!(model, scene, roadway, 1)
    @test model.driver.mlane.rec.nscenes == 1
end

function test_delayed_driver_simulation()
    # set up a scene with two vehicles in a following pattern
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

    # the idea of this test is to confirm that using a delayed driver model 
    # with a 0.0 reaction time gives identical results to not using a delay
    normal_scene = copy!(Scene(num_veh), scene)
    without_delay_scene = copy!(Scene(num_veh), scene)
    with_delay_scene = copy!(Scene(num_veh), scene)

    # normal
    models = Dict{Int, DriverModel}()
    models[1] = Tim2DDriver(context)
    models[2] = Tim2DDriver(context)
    simulate!(normal_scene, models, roadway, rec, T)
    
    # without delay
    models = Dict{Int, DriverModel}()
    models[1] = DelayedDriver(Tim2DDriver(context), reaction_time = 0.0)
    models[2] = DelayedDriver(Tim2DDriver(context), reaction_time = 0.0)
    simulate!(without_delay_scene, models, roadway, rec, T)

    @test normal_scene == without_delay_scene

    # then run with delay and confirm they are different
    models = Dict{Int, DriverModel}()
    models[1] = DelayedDriver(Tim2DDriver(
        context, mlane = MOBIL(context)), reaction_time = 0.1)
    models[2] = DelayedDriver(Tim2DDriver(
        context, mlane = MOBIL(context)), reaction_time = 0.1)
    simulate!(with_delay_scene, models, roadway, rec, T)

    @test with_delay_scene != normal_scene
end

@time test_delayed_driver_observe()
@time test_delayed_driver_simulation()
