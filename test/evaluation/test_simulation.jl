# using Base.Test
# using AutomotiveDrivingModels

# push!(LOAD_PATH, "../../src")
# include("../../src/evaluation/simulation.jl")
# include("../../src/utils/automotive.jl")

function test_simulate()
    # set up and sim a scene with two vehicles w/ constant vel
    context = IntegratedContinuous(.1, 1)
    num_veh = 2
    
    models = Dict{Int, DriverModel}()
    mlon = StaticLongitudinalDriver(0.)
    models[1] = Tim2DDriver(context, mlon = mlon)
    models[2] = Tim2DDriver(context, mlon = mlon)

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

    simulate!(scene, models, roadway, rec, T)

    # check vehicle states
    veh_1 = scene[get_index_of_first_vehicle_with_id(scene, 1)]
    @test veh_1.state.v == base_speed
    @test veh_1.state.posG.x ≈ 11.
    @test veh_1.state.posG.y ≈ 0.0
    @test veh_1.state.posF.t ≈ 0.0
    @test veh_1.state.posF.s ≈ 11.
    @test veh_1.state.posF.ϕ ≈ 0.

    veh_2 = scene[get_index_of_first_vehicle_with_id(scene, 2)]
    @test veh_2.state.v == base_speed
    @test veh_2.state.posG.x ≈ 1.
    @test veh_2.state.posG.y ≈ 0.0
    @test veh_2.state.posF.t ≈ 0.0
    @test veh_2.state.posF.s ≈ 1.
    @test veh_2.state.posF.ϕ ≈ 0.

    # check record
    @test rec.nscenes == 10
    rec_scene = get_scene(rec, 0)
    @test rec_scene == scene
    @test convert(Float64, get(SPEED, rec, roadway, 1, 0)) ≈ 1.0
    @test convert(Float64, get(SPEED, rec, roadway, 2, 0)) ≈ 1.0
    @test convert(Float64, get(ACC, rec, roadway, 1, 0)) ≈ 0.0
    @test convert(Float64, get(ACC, rec, roadway, 2, 0)) ≈ 0.0
end

@time test_simulate()