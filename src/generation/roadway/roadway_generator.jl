export 
    RoadwayGenerator,
    StaticRoadwayGenerator,
    rand!

"""
# Description: 
    - RoadwayGenerator is the abstract type underlying the roadway generators.
"""
abstract RoadwayGenerator

Base.rand!(gen::RoadwayGenerator, roadway::Roadway, seed::Int64) = error(
    "rand! not implemented for $(scene_generator)")

"""
# Description:
    - StaticRoadwayGenerator has a single roadway that it always returns.
"""
type StaticRoadwayGenerator <: RoadwayGenerator
    roadway::Roadway
end

Base.rand!(gen::StaticRoadwayGenerator, roadway::Roadway, seed::Int64) = gen.roadway
Base.rand!(gen::StaticRoadwayGenerator, seed::Int64) = gen.roadway
