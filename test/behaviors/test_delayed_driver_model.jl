# using Base.Test
# using AutoRisk

function test_delayed_driver_observe()
    roadway = gen_straight_roadway(1)
    num_veh = 2
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

    # test without reaction time
    model = DelayedDriver(Tim2DDriver(.1), reaction_time = 0.0)
    observe!(model, scene, roadway, 1)
    @test model.rec.nframes == 1
    @test length(model.rec.frames) == 1
    @test model.driver.mlane.rec.nframes == 1

    # test with reaction time
    model = DelayedDriver(Tim2DDriver(.1), reaction_time = 0.1)
    observe!(model, scene, roadway, 1)
    @test model.rec.nframes == 1
    @test length(model.rec.frames) == 2
    @test model.driver.mlane.rec.nframes == 0
    observe!(model, scene, roadway, 1)
    @test model.driver.mlane.rec.nframes == 1
end

function test_delayed_driver_simulation()
    # set up a scene with two vehicles in a following pattern
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

    # the idea of this test is to confirm that using a delayed driver model 
    # with a 0.0 reaction time gives identical results to not using a delay
    normal_scene = copyto!(Scene(num_veh), scene)
    without_delay_scene = copyto!(Scene(num_veh), scene)
    with_delay_scene = copyto!(Scene(num_veh), scene)

    # normal
    models = Dict{Int, DriverModel}()
    models[1] = Tim2DDriver(.1)
    models[2] = Tim2DDriver(.1)
    simulate!(LatLonAccel, rec, normal_scene, roadway, models, T)
    
    # without delay
    models = Dict{Int, DriverModel}()
    models[1] = DelayedDriver(Tim2DDriver(.1), reaction_time = 0.0)
    models[2] = DelayedDriver(Tim2DDriver(.1), reaction_time = 0.0)
    simulate!(LatLonAccel, rec, without_delay_scene, roadway, models, T)

    @test normal_scene == without_delay_scene

    # then run with delay and confirm they are different
    models = Dict{Int, DriverModel}()
    models[1] = DelayedDriver(Tim2DDriver(
        .1, mlane = MOBIL(.1)), reaction_time = 0.1)
    models[2] = DelayedDriver(Tim2DDriver(
        .1, mlane = MOBIL(.1)), reaction_time = 0.1)
    simulate!(LatLonAccel, rec, with_delay_scene, roadway, models, T)

    @test with_delay_scene != normal_scene
end

@time test_delayed_driver_observe()
@time test_delayed_driver_simulation()
