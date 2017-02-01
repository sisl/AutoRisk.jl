export 
    RoadwayGenerator,
    StaticRoadwayGenerator,
    reset!

"""
# Description: 
    - RoadwayGenerator is the abstract type underlying the roadway generators.
"""
abstract RoadwayGenerator

reset!(gen::RoadwayGenerator, roadway::Roadway, seed::Int64) = error(
    "reset! not implemented for $(scene_generator)")

"""
# Description:
    - StaticRoadwayGenerator has a single roadway that it always returns.
"""
type StaticRoadwayGenerator <: RoadwayGenerator
    roadway::Roadway
end

reset!(gen::StaticRoadwayGenerator, roadway::Roadway, seed::Int64) = gen.roadway
reset!(gen::StaticRoadwayGenerator, seed::Int64) = gen.roadway
