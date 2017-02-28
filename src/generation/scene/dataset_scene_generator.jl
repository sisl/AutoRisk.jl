export 
    DatasetSceneGenerator,
    rand!

type DatasetSceneGenerator <: SceneGenerator
    trajdata::Trajdata
    veh_ids::Vector{Int64}
    next_idx::Int64
    rng::MersenneTwister
end

function Base.rand!(gen::DatasetSceneGenerator, scene::Scene, roadway::Roadway, 
        seed::Int64)
    # remove old contents of scene and models
    empty!(scene)

    # get the first scene with the next veh_id and then increment
    frame_idx = get_first_frame_with_id(gen.trajdata, gen.veh_ids[gen.next_idx])
    get!(scene, gen.trajdata, frame_idx)
    gen.next_idx += 1
end