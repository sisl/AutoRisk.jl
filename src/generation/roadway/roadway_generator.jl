export 
    RoadwayGenerator,
    StaticRoadwayGenerator,
    rand!

"""
# Description: 
    - RoadwayGenerator is the abstract type underlying the roadway generators.
"""
abstract type RoadwayGenerator end

Random.rand!(gen::RoadwayGenerator, roadway::Roadway, seed::Int64) = error(
    "rand! not implemented for $(scene_generator)")

"""
# Description:
    - StaticRoadwayGenerator has a single roadway that it always returns.
"""
type StaticRoadwayGenerator <: RoadwayGenerator
    roadway::Roadway
end

Random.rand!(gen::StaticRoadwayGenerator, roadway::Roadway, seed::Int64) = gen.roadway
Random.rand!(gen::StaticRoadwayGenerator, seed::Int64) = gen.roadway
