using Base.Test
using AutoRisk

function test_inverse_ttc_to_ttc()
    # missing
    inv_ttc = FeatureValue(0.0, FeatureState.MISSING)
    ttc = inverse_ttc_to_ttc(inv_ttc)
    @test ttc.i == FeatureState.MISSING
    @test ttc.v == 30.0

    # pulling away
    inv_ttc = FeatureValue(0.0, FeatureState.GOOD)
    ttc = inverse_ttc_to_ttc(inv_ttc, censor_hi = 30.)
    @test ttc.i == FeatureState.CENSORED_HI
    @test ttc.v == 30.0

    # collision
    value = 10.0
    inv_ttc = FeatureValue(value, FeatureState.CENSORED_HI)
    ttc = inverse_ttc_to_ttc(inv_ttc)
    @test ttc.i == FeatureState.GOOD
    @test ttc.v == 1. / value
end

function test_push_forward_records_scene_record()
    max_n_scenes = 3
    rec = SceneRecord(max_n_scenes, 1., 1)
    carcount = 0
    for i in max_n_scenes:-1:1
        scene = Scene(1)
        push!(scene, Vehicle(VehicleState(), VehicleDef(), i))
        update!(rec, scene)
    end
    pastframe = -1
    push_forward_records!(rec, pastframe)

    @test rec.frames[1].entities[1].id == 2
    @test rec.frames[2].entities[1].id == 3
    @test rec.nframes == 2
end

function test_executed_hard_brake()
    roadway = gen_straight_roadway(1, 500.)
    max_n_scenes = 3
    rec = SceneRecord(max_n_scenes, .1, 1)
    scene = Scene()
    state = VehicleState(VecSE2(), 10.)
    push!(scene, Vehicle(state, VehicleDef(), 1))
    update!(rec, scene)
    scene = Scene()
    state = VehicleState(VecSE2(), 9.)
    push!(scene, Vehicle(state, VehicleDef(), 1))
    update!(rec, scene)
    scene = Scene()
    state = VehicleState(VecSE2(), 8.)
    push!(scene, Vehicle(state, VehicleDef(), 1))
    update!(rec, scene)

    # did execute hard brake
    executed = executed_hard_brake(
        rec, roadway, 1, hard_brake_threshold = -4., n_past_frames = 2)
    @test executed == true

    # did not
    scene = Scene()
    state = VehicleState(VecSE2(), 8.)
    push!(scene, Vehicle(state, VehicleDef(), 1))
    update!(rec, scene)
    executed = executed_hard_brake(
        rec, roadway, 1, hard_brake_threshold = -4., n_past_frames = 2)
    @test executed == false

end

@time test_inverse_ttc_to_ttc()
@time test_push_forward_records_scene_record()
@time test_executed_hard_brake()