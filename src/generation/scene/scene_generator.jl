export 
    SceneGenerator,
    rand!

"""
# Description:
    - SceneGenerator is the abstract type underlying the scene generators.
"""
abstract SceneGenerator

Base.rand!(scene_generator::SceneGenerator, scene::Scene, roadway::Roadway, 
    seed::Int64) = error("rand! not implemented for $(scene_generator)")

pdf(scene_generator::SceneGenerator, scene::Scene) = 1.
logpdf(scene_generator::SceneGenerator, scene::Scene) = 0.