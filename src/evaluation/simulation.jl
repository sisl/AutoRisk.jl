export 
    simulate!

function correct_vehicle_states(
        scene::Scene; 
        ϕ_low::Float64 = -pi/2, 
        ϕ_high::Float64 = pi/2
    )

    for (i, vehicle) in enumerate(scene)
        if !(ϕ_low < vehicle.state.posF.ϕ < ϕ_high)

            new_state = VehicleState(
                    vehicle.state.posG, 
                    Frenet(
                        vehicle.state.posF.roadind,
                        vehicle.state.posF.s,
                        vehicle.state.posF.t,
                        clamp(vehicle.state.posF.ϕ, ϕ_low, ϕ_high), 
                    ), 
                    vehicle.state.v
                )
            new_vehicle = Vehicle(new_state, vehicle.def, vehicle.id)
            scene[i] = new_vehicle
        end
    end
end

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
function simulate!(
        ::Type{A},
        rec::EntityQueueRecord{S,D,I}, 
        scene::EntityFrame{S,D,I}, 
        roadway::R,
        models::Dict{I,M}, 
        T::Float64;
        update_first_scene::Bool = true
    ) where {S,D,I,A,R,M<:DriverModel}
    if update_first_scene
        update!(rec, scene)
    end
    actions = Array{A}(undef, length(scene))
    for t in 0:rec.timestep:(T - rec.timestep)
        get_actions!(actions, scene, roadway, models)
        tick!(scene, roadway, actions, rec.timestep)
        correct_vehicle_states(scene)
        update!(rec, scene)
    end
    return rec
end
