# using Base.Test
# using AutoRisk

function debug_scene_gen()
    gen = DebugSceneGenerator(
        lo_Δs = 5.,
        hi_Δs = 5.,
        lo_v_rear = 1.,
        hi_v_rear = 1.,
        lo_v_fore = 0.,
        hi_v_fore = 0.
    )
    roadway = gen_straight_roadway(1)
    scene = rand!(gen, Scene(2), roadway, 1)
    
    rear = scene.vehicles[1]
    fore = scene.vehicles[2]
    @test rear.state.v == 1.
    @test fore.state.v == 0.
end

@time debug_scene_gen()
