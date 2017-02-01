export 
    simulate!

"""
# Description:
    - Simulate a scene for a period of time.

# Args:
    - scene: scene to simulate
    - models: driver models to use for deriving actions
    - roadway: roadway on which to simulate
    - rec: record where to store scenes
    - T: time for which to simulate
"""
function simulate!(scene::Scene, models::Dict{Int, DriverModel},
        roadway::Roadway, rec::SceneRecord, T::Float64)
    actions = Array(DriveAction, length(scene))

    # simulate for T seconds in rec.timestep-length substeps
    for t in 0:rec.timestep:(T - rec.timestep)
        get_actions!(actions, scene, roadway, models)
        tick!(scene, roadway, actions, models)
        update!(rec, scene)
    end
    return rec
end