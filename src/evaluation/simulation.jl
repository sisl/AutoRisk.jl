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
function simulate!{S,D,I,A,R,M<:DriverModel}(
    ::Type{A},
    rec::EntityQueueRecord{S,D,I}, 
    scene::EntityFrame{S,D,I}, 
    roadway::R,
    models::Dict{I,M}, 
    T::Float64
    )
    actions = Array(A, length(scene))
    for t in 0:rec.timestep:(T - rec.timestep)
        get_actions!(actions, scene, roadway, models)
        tick!(scene, roadway, actions, rec.timestep)
        update!(rec, scene)
    end
    return rec
end
