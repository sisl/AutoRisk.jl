# using AutoRisk

# include("../../../scripts/collection/collect_heuristic_dataset.jl")
# include("../../../scripts/collection/heuristic_dataset_config.jl")

#=
Test MOBIL when exiting a curve:
    - one MOBIL vehicle in the inner lane about to exit a curve
    - one static vehicle slower than MOBIL in front in the same lane
    - one static vehicle at the same speed traveling slight behind in 
        the other lane
=#
function test_mobil_when_exiting_curve()
    num_veh = 3
    # two-lane stadium
    roadway = gen_stadium_roadway(2, length = 400.0, radius = 100.0)

    # build the scene
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
    models = Dict{Int, DriverModel}()
    mlon = DelayedIntelligentDriverModel(.1, t_d = 0.0)
    models[1] = Tim2DDriver(.1, mlon = mlon)
    models[2] = Tim2DDriver(.1, mlon = mlon)
    simulate!(LatLonAccel, rec, no_delay_init_scene, roadway, models, T)
end

@time test_mobil_when_exiting_curve()