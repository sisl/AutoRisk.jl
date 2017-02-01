export 
    SceneGenerator

"""
# Description:
    - SceneGenerator is the abstract type underlying the scene generators.
"""
abstract SceneGenerator

reset!(scene_generator::SceneGenerator, scene::Scene, roadway::Roadway, 
    seed::Int64) = error("reset! not implemented for $(scene_generator)")