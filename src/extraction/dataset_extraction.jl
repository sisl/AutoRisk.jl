export
    extract_vehicle_frame_targets!,
    extract_frame_targets!,
    extract_targets!,
    extract_vehicle_features!,
    extract_features!

#=
Notes:
1. trajdata frames are stored with the last frame last, so 
    get!(scene, trajdata, 1) gets the first frame and 
    get!(scene, trajdata, last_frame_idx) gets the last frame added.
2. scene records are the opposite in the sense that in calling update! 
    on a scene record, you want to do so in the order the scenes occurred 
    so that the most recent scene is stored first in the record.
    - scene records must also have multiple frames loaded into them to 
        compute certain features (e.g., accel and velocity)
    - pastframe indexing is with negative values
=#

##################### Target Extraction #####################

"""
# Description:
    - Extract targets for a single vehicle in a single frame.

# Args:
    - rec: scene rec from which to derive features
    - roadway: roadway in state
    - targets: array in which to insert features
        shape = (target_dim, num_vehicles)
    - veh_idx: the index of the vehicle in the scene
    - target_idx: the index in the targets array corresponding to this vehicle
    - pastframe: frame index in the past to use
"""
function extract_vehicle_frame_targets!(rec::SceneRecord, roadway::Roadway, 
        targets::Array{Float64}, veh_idx::Int64, target_idx::Int64, 
        pastframe::Int64)

    ### collision targets
    # types: 
    # 0 = no collision 
    # 1 = lane-change
    # 2 = rear-end-lead
    # 3 = rear-end-follow
    collision_type = get_collision_type(rec, roadway, veh_idx, pastframe)
    in_collision = collision_type > 0 ? true : false
    if in_collision
        targets[collision_type, target_idx] = 1.
    end

    ### behavioral targets
    # hard brake
    hard_brake = executed_hard_brake(rec, roadway, veh_idx, pastframe, 
        hard_brake_threshold = -3., n_past_frames = 2)
    if hard_brake
        targets[4, target_idx] = 1.
    end

    ### conflict targets
    # time to collision
    inv_ttc = get(INV_TTC, rec, roadway, veh_idx, pastframe)
    ttc = inverse_ttc_to_ttc(inv_ttc, censor_hi = 30.0)
    if ttc.i != FeatureState.MISSING && ttc.v < 3.
        targets[5, target_idx] = 1.
    end

    return in_collision
end

"""
# Description:
    - Extract targets for all vehicles in the pastframe

# Args:
    - rec: scene rec from which to derive features
    - roadway: roadway in state
    - targets: array in which to insert features
        shape = (target_dim, num_vehicles)
    - veh_id_to_idx: dictionary mapping vehicle ids to indices in the scene
        this is only useful if the veh_idx in the scene cannot change, which
        is only the case is vehicles cannot leave the scene
    - veh_idx_can_change: whether veh_idx can change across scenes in the rec
    - done: the set of vehicle ids that have either already left the scene or 
        already collided
    - pastframe: the (negative) index of the frame in the record to extract 
        targets for
"""
function extract_frame_targets!(rec::SceneRecord, roadway::Roadway, 
        targets::Array{Float64}, veh_id_to_idx::Dict{Int,Int}, 
        veh_idx_can_change::Bool, done::Set{Int}, pastframe::Int64)

    scene = rec[pastframe]
    for (veh_id, orig_veh_idx) in veh_id_to_idx
        # track which vehicles have left the scene or have already 
        # collided and skip them once either has occurred
        if in(veh_id, done)
            continue
        end
        in_collision = false
        in_scene = true

        # separate the cases where the vehicle index can change,
        # because if it cannot we can save a lot of time by not 
        # recomputing it for each frame
        if veh_idx_can_change
            veh_idx = findfirst(scene, veh_id)
            # if the vehicle left the scene, then we assume that 
            # it will not reenter and add it to the done set
            in_scene = veh_idx == 0 ? false : true
        else
            veh_idx = orig_veh_idx
        end

        # extract target values for this vehicle in the current 
        # frame, provided it has not left the scene
        if in_scene
            in_collision = extract_vehicle_frame_targets!(rec, roadway, targets,
                veh_idx, orig_veh_idx, pastframe)
        end

        # if the vehicle has left the scene or been in a collision
        # then stop computing target values for it 
        if !in_scene || in_collision
            push!(done, veh_id)
        end

    end
end

"""
# Description:
    - Extract target values for every vehicle in the record across all 
        scenes in that record.

# Args:
    - rec: scene rec from which to derive features
    - roadway: roadway in state
    - targets: array in which to insert features
        shape = (target_dim, num_vehicles)
    - veh_id_to_idx: dictionary mapping vehicle ids to indices in the scene
        this is only useful if the veh_idx in the scene cannot change, which
        is only the case is vehicles cannot leave the scene
    - veh_idx_can_change: whether veh_idx can change across scenes in the rec
    - start_frame: at which frame in the rec to begin target collection
        defaults to the first
    - done: set used to track which vehicles have either left the scene or 
        collided (and )
"""
function extract_targets!(rec::SceneRecord, roadway::Roadway, 
        targets::Array{Float64}, veh_id_to_idx::Dict{Int, Int},
        veh_idx_can_change::Bool, start_frame::Int64 = rec.nframes - 1;
        done::Set{Int64} = Set{Int64}())
    # reset targets container and done set
    fill!(targets, 0)
    empty!(done)

    # move forward in time through each scene in the record, 
    # computing target values for every car in veh_id_to_idx
    # at each frame
    for pastframe in -start_frame:0
        extract_frame_targets!(rec, roadway, targets, veh_id_to_idx, 
            veh_idx_can_change, done, pastframe)
    end
end

##################### Feature Extraction #####################

function pull_features!(ext::AbstractFeatureExtractor, rec::SceneRecord, 
        roadway::Roadway, models::Dict{Int, DriverModel}, features::Array{Float64},
        steps::Int64 = 1)
    # reset features container
    fill!(features, 0)
    # extract features for each vehicle in the scene for each timestep 
    # inserting into features in past to present order
    for t in 1:steps
        for (vidx, veh) in enumerate(rec[-(t - 1)])
            features[:, steps - t + 1, vidx] = pull_features!(
                ext, rec, roadway, vidx, models, -(t-1))
        end
    end
    return features
end