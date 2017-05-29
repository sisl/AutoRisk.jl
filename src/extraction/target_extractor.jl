export
    TargetExtractor,
    length,
    feature_names,
    pull_features!,
    extract_frame_targets!,
    extract_targets!

type TargetExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    hard_brake_threshold::Float64
    hard_brake_n_past_frames::Int
    ttc_threshold::Float64
    function TargetExtractor(;
            hard_brake_threshold::Float64 = -3.,
            hard_brake_n_past_frames::Int = 2,
            ttc_threshold::Float64 = 3.
        )
        num_features = 5
        return new(
            zeros(Float64, num_features), 
            num_features,
            hard_brake_threshold,
            hard_brake_n_past_frames,
            ttc_threshold
        )
    end
end
Base.length(ext::TargetExtractor) = ext.num_features
function feature_names(ext::TargetExtractor)
    return String[
        "lane_change_collision",
        "rear_end_collision_ego_vehicle_in_front",
        "rear_end_collision_ego_vehicle_in_back",
        "hard_brake",
        "time_to_collision_conflict"
    ]
end
function AutomotiveDrivingModels.pull_features!(
        ext::TargetExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,
        pastframe::Int = 0
    )
    fill!(ext.features, 0)

    collision_type = get_collision_type(rec, roadway, veh_idx, pastframe)
    in_collision = collision_type > 0 ? true : false
    if in_collision
        ext.features[collision_type] = 1.
    end

    hard_brake = executed_hard_brake(rec, roadway, veh_idx, pastframe, 
        hard_brake_threshold = ext.hard_brake_threshold, 
        n_past_frames = ext.hard_brake_n_past_frames)
    if hard_brake
        ext.features[4] = 1.
    end

    ### conflict targets
    # time to collision
    inv_ttc = get(INV_TTC, rec, roadway, veh_idx, pastframe)
    ttc = inverse_ttc_to_ttc(inv_ttc, censor_hi = 30.0)
    if ttc.i != FeatureState.MISSING && ttc.v < ext.ttc_threshold
        ext.features[5] = 1.
    end

    return ext.features
end

"""
# Description:
    - Extract targets for all vehicles in the pastframe

# Args:
    - ext: target extractor
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
function extract_frame_targets!(
        ext::AbstractFeatureExtractor,
        rec::SceneRecord, 
        roadway::Roadway, 
        targets::Array{Float64}, 
        veh_id_to_idx::Dict{Int,Int}, 
        veh_idx_can_change::Bool, 
        done::Set{Int}, 
        pastframe::Int64
    )

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
            targets[:, orig_veh_idx] = pull_features!(ext, rec, 
                roadway, veh_idx, pastframe)
            in_collision = any(targets[1:3, orig_veh_idx] .> 0)
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
    - ext: target extractor
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
function extract_targets!(
        ext::AbstractFeatureExtractor,
        rec::SceneRecord, 
        roadway::Roadway, 
        targets::Array{Float64}, 
        veh_id_to_idx::Dict{Int, Int},
        veh_idx_can_change::Bool, 
        start_frame::Int64 = rec.nframes - 1;
        done::Set{Int64} = Set{Int64}()
    )
    # reset targets container and done set
    fill!(targets, 0)
    empty!(done)

    # move forward in time through each scene in the record, 
    # computing target values for every car in veh_id_to_idx
    # at each frame
    for pastframe in -start_frame:0
        extract_frame_targets!(ext, rec, roadway, targets, veh_id_to_idx, 
            veh_idx_can_change, done, pastframe)
    end
end
