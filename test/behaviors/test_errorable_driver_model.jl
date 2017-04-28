# using Base.Test
# using AutoRisk

function test_errorable_driver_observe()
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
    model = ErrorableDriverModel(Tim2DDriver(.1), 
            p_a_to_i = 0.0, 
            p_i_to_a = 0.0)
    observe!(model, scene, roadway, 1)
    @test model.is_attentive == true

    # test with reaction time
    model = ErrorableDriverModel(Tim2DDriver(.1),
            p_a_to_i = 1.0, 
            p_i_to_a = 0.0)

    observe!(model, scene, roadway, 1)
    @test model.is_attentive == false
end


@time test_errorable_driver_observe()
