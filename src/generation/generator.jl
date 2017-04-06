export 
    Generator,
    FactoredGenerator,
    rand!

abstract Generator

type FactoredGenerator <: Generator
    roadway_gen::RoadwayGenerator
    scene_gen::SceneGenerator
    behavior_gen::BehaviorGenerator
end

function Base.rand!(gen::FactoredGenerator, roadway::Roadway, scene::Scene, 
        models::Dict{Int,DriverModel}, seed::Int64)
    rand!(gen.roadway_gen, roadway, seed)
    rand!(gen.scene_gen, scene, roadway, seed)
    rand!(gen.behavior_gen, models, scene, seed)
end