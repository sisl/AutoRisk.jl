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
    if ttc.i != FeatureState.MISSING && ttc.v < 2.5
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

    scene = get_scene(rec, pastframe)
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
            veh_idx = get_index_of_first_vehicle_with_id(scene, veh_id)
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
        veh_idx_can_change::Bool, start_frame::Int64 = rec.nscenes - 1;
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

# """
# # Description:
#     - Extract features for a single vehicle

# # Args:
#     - rec: scene rec from which to derive features
#     - roadway: roadway in state
#     - models: models for all drivers in the rec (id => model)
#     - features: array in which to insert features
#         shape = (feature_dim, num_vehicles)
#     - veh_id: id of the vehicle for which to derive features
#     - veh_idx: index in the _most recent_ scene of the vehicle 
#         with id veh_id. Note that if features need to be derived 
#         from scenes earlier than the current, this must be recomputed.
# """
# function extract_vehicle_features!(rec::SceneRecord, roadway::Roadway, 
#         models::Dict{Int64, DriverModel}, features::Array{Float64},
#         veh_id::Int64, veh_idx::Int64)

#     # extract scene features
#     scene = get_scene(rec, 0)

#     # ego vehicle features
#     veh_ego = scene[veh_idx]
#     features[1, veh_idx] = veh_ego.state.posF.t
#     features[2, veh_idx] = veh_ego.state.posF.ϕ
#     features[3, veh_idx] = veh_ego.state.v
#     features[4, veh_idx] = veh_ego.def.length
#     features[5, veh_idx] = veh_ego.def.width
#     features[6, veh_idx] = convert(Float64, 
#         get(ACC, rec, roadway, veh_idx))
#     features[7, veh_idx] = convert(Float64, 
#         get(JERK, rec, roadway, veh_idx))
#     features[8, veh_idx] = convert(Float64, 
#         get(TURNRATEG, rec, roadway, veh_idx))
#     features[9, veh_idx] = convert(Float64, 
#         get(ANGULARRATEG, rec, roadway, veh_idx))
#     features[10, veh_idx] = convert(Float64, 
#         get(TURNRATEF, rec, roadway, veh_idx))
#     features[11, veh_idx] = convert(Float64, 
#         get(ANGULARRATEF, rec, roadway, veh_idx))
#     features[12, veh_idx] = convert(Float64, 
#         get(LANECURVATURE, rec, roadway, veh_idx))
#     features[13, veh_idx] = convert(Float64, 
#         get(MARKERDIST_LEFT, rec, roadway, veh_idx))
#     features[14, veh_idx] = convert(Float64, 
#         get(MARKERDIST_RIGHT, rec, roadway, veh_idx))
#     features[15, veh_idx] = convert(Float64, 
#         features[13, veh_idx] < -1.0 || features[14] < -1.0)
#     features[16, veh_idx] = convert(Float64, 
#         veh_ego.state.v < 0.0)
#     set_dual_feature!(features, 17, 
#         get(LANEOFFSETLEFT, rec, roadway, veh_idx), veh_idx)
#     set_dual_feature!(features, 19, 
#         get(LANEOFFSETRIGHT, rec, roadway, veh_idx), veh_idx)

#     # lane features
#     features[21, veh_idx] = convert(Float64, 
#         get(HAS_LANE_LEFT, rec, roadway, veh_idx))
#     features[22, veh_idx] = convert(Float64, 
#         get(HAS_LANE_RIGHT, rec, roadway, veh_idx))

#     # do not pass the later-calculated neighbor to these functions 
#     # because they assume that the NeighborLongitudinalResult
#     # has not yet had the lengths of the vehicles substracted from
#     # the distance between the vehicles

#     # timegap is the time between when this vehicle's front bumper
#     # will be in the position currently occupied by the vehicle 
#     # infront's back bumper
#     set_dual_feature!(features, 23, 
#         get(TIMEGAP, rec, roadway, veh_idx, censor_hi = 30.0), veh_idx)

#     # inverse time to collision is the time until a collision 
#     # assuming that no actions are taken
#     # inverse is taken so as to avoid infinite value, so flip here to get back
#     # to TTC
#     inv_ttc = get(INV_TTC, rec, roadway, veh_idx)
#     ttc = inverse_ttc_to_ttc(inv_ttc, censor_hi = 30.0)
#     set_dual_feature!(features, 25, ttc, veh_idx)

#     # feature for whether a collision has already occurred
#     features[27, veh_idx] = convert(Float64, 
#         get_collision_type(rec, roadway, veh_idx))

#     # neighbor features
#     F = VehicleTargetPointFront()
#     R = VehicleTargetPointRear()
#     fore_M = get_neighbor_fore_along_lane(
#         scene, veh_idx, roadway, F, R, F)
#     fore_L = get_neighbor_fore_along_left_lane(
#         scene, veh_idx, roadway, F, R, F)
#     fore_R = get_neighbor_fore_along_right_lane(
#         scene, veh_idx, roadway, F, R, F)
#     rear_M = get_neighbor_rear_along_lane(
#         scene, veh_idx, roadway, R, F, R)
#     rear_L = get_neighbor_rear_along_left_lane(
#         scene, veh_idx, roadway, R, F, R)
#     rear_R = get_neighbor_rear_along_right_lane(
#         scene, veh_idx, roadway, R, F, R)
#     if fore_M.ind != 0
#         fore_fore_M = get_neighbor_fore_along_lane(
#             scene, fore_M.ind, roadway, F, R, F)
#     else
#         fore_fore_M = NeighborLongitudinalResult(0, 0.)
#     end

#     next_feature_idx = 28
#     set_neighbor_features!(
#         features, next_feature_idx, fore_M, scene, rec, roadway, veh_idx)
#     set_neighbor_features!(
#         features, next_feature_idx += 5, fore_L, scene, rec, roadway, veh_idx)
#     set_neighbor_features!(
#         features, next_feature_idx += 5, fore_R, scene, rec, roadway, veh_idx)
#     set_neighbor_features!(
#         features, next_feature_idx += 5, rear_M, scene, rec, roadway, veh_idx)
#     set_neighbor_features!(
#         features, next_feature_idx += 5, rear_L, scene, rec, roadway, veh_idx)
#     set_neighbor_features!(
#         features, next_feature_idx += 5, rear_R, scene, rec, roadway, veh_idx)
#     set_neighbor_features!(
#         features, next_feature_idx += 5, fore_fore_M, scene, rec, roadway, veh_idx)

#     # extract driver behavior features for ego and surrounding vehicles
#     idxs::Vector{Int64} = [veh_idx, fore_M.ind, fore_L.ind, fore_R.ind, 
#         rear_M.ind, rear_L.ind, rear_R.ind, fore_fore_M.ind]
#     next_feature_idx = 62
#     for veh_index in idxs
#         next_feature_idx = set_behavioral_features(scene, models, features, 
#             veh_index, next_feature_idx)
#     end 
# end

# """
# # Description:
#     - Extract features for all vehicles in the scene record.

# # Args:
#     - rec: scene rec from which to derive features
#     - roadway: roadway in state
#     - models: models for all drivers in the rec (id => model)
#     - features: array in which to insert features
#         shape = (feature_dim, num_vehicles)
#     - done: a set of vehicle ids for which features should not be extracted
# """
# function extract_features!(rec::SceneRecord, roadway::Roadway, 
#         models::Dict{Int, DriverModel}, features::Array{Float64}; 
#         done::Set{Int} = Set{Int}())

#     # reset features container
#     fill!(features, 0)

#     # extract features for each vehicle in the scene
#     for (vidx, veh) in enumerate(get_scene(rec, 0))
#         if !in(veh.def.id, done)
#             extract_vehicle_features!(rec, roadway, models, features, 
#                 veh.def.id, vidx)
#         end
#     end
# end

function extract_features!(ext::AbstractFeatureExtractor, rec::SceneRecord, 
        roadway::Roadway, models::Dict{Int, DriverModel}, features::Array{Float64})
    # reset features container
    fill!(features, 0)
    # extract features for each vehicle in the scene
    for (vidx, veh) in enumerate(get_scene(rec, 0))
        pull_features!(ext, features, rec, roadway, vidx, models)
    end
end