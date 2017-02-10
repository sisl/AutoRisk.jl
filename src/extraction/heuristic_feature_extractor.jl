export 
    HeuristicFeatureExtractor,
    length,
    pull_features!

type HeuristicFeatureExtractor <: AbstractFeatureExtractor
end
Base.length(ext::HeuristicFeatureExtractor) = 166
function AutomotiveDrivingModels.pull_features!(
        ext::HeuristicFeatureExtractor,
        features::Array{Float64},
        rec::SceneRecord,
        roadway::Roadway,
        veh_idx::Int,
        models::Dict{Int, DriverModel})
    # extract scene features
    scene = get_scene(rec, 0)
    veh_ego = scene[veh_idx]

    # ego vehicle features
    features[1, veh_idx] = veh_ego.state.posF.t
    features[2, veh_idx] = veh_ego.state.posF.ϕ
    features[3, veh_idx] = veh_ego.state.v
    features[4, veh_idx] = veh_ego.def.length
    features[5, veh_idx] = veh_ego.def.width
    features[6, veh_idx] = convert(Float64, 
        get(ACC, rec, roadway, veh_idx))
    features[7, veh_idx] = convert(Float64, 
        get(JERK, rec, roadway, veh_idx))
    features[8, veh_idx] = convert(Float64, 
        get(TURNRATEG, rec, roadway, veh_idx))
    features[9, veh_idx] = convert(Float64, 
        get(ANGULARRATEG, rec, roadway, veh_idx))
    features[10, veh_idx] = convert(Float64, 
        get(TURNRATEF, rec, roadway, veh_idx))
    features[11, veh_idx] = convert(Float64, 
        get(ANGULARRATEF, rec, roadway, veh_idx))
    features[12, veh_idx] = convert(Float64, 
        get(LANECURVATURE, rec, roadway, veh_idx))
    features[13, veh_idx] = convert(Float64, 
        get(MARKERDIST_LEFT, rec, roadway, veh_idx))
    features[14, veh_idx] = convert(Float64, 
        get(MARKERDIST_RIGHT, rec, roadway, veh_idx))
    features[15, veh_idx] = convert(Float64, 
        features[13, veh_idx] < -1.0 || features[14] < -1.0)
    features[16, veh_idx] = convert(Float64, 
        veh_ego.state.v < 0.0)
    set_dual_feature!(features, 17, 
        get(LANEOFFSETLEFT, rec, roadway, veh_idx), veh_idx)
    set_dual_feature!(features, 19, 
        get(LANEOFFSETRIGHT, rec, roadway, veh_idx), veh_idx)

    # lane features
    features[21, veh_idx] = convert(Float64, 
        get(HAS_LANE_LEFT, rec, roadway, veh_idx))
    features[22, veh_idx] = convert(Float64, 
        get(HAS_LANE_RIGHT, rec, roadway, veh_idx))

    # do not pass the later-calculated neighbor to these functions 
    # because they assume that the NeighborLongitudinalResult
    # has not yet had the lengths of the vehicles substracted from
    # the distance between the vehicles

    # timegap is the time between when this vehicle's front bumper
    # will be in the position currently occupied by the vehicle 
    # infront's back bumper
    set_dual_feature!(features, 23, 
        get(TIMEGAP, rec, roadway, veh_idx, censor_hi = 30.0), veh_idx)

    # inverse time to collision is the time until a collision 
    # assuming that no actions are taken
    # inverse is taken so as to avoid infinite value, so flip here to get back
    # to TTC
    inv_ttc = get(INV_TTC, rec, roadway, veh_idx)
    ttc = inverse_ttc_to_ttc(inv_ttc, censor_hi = 30.0)
    set_dual_feature!(features, 25, ttc, veh_idx)

    # feature for whether a collision has already occurred
    features[27, veh_idx] = convert(Float64, 
        get_collision_type(rec, roadway, veh_idx))

    # neighbor features
    F = VehicleTargetPointFront()
    R = VehicleTargetPointRear()
    fore_M = get_neighbor_fore_along_lane(
        scene, veh_idx, roadway, F, R, F)
    fore_L = get_neighbor_fore_along_left_lane(
        scene, veh_idx, roadway, F, R, F)
    fore_R = get_neighbor_fore_along_right_lane(
        scene, veh_idx, roadway, F, R, F)
    rear_M = get_neighbor_rear_along_lane(
        scene, veh_idx, roadway, R, F, R)
    rear_L = get_neighbor_rear_along_left_lane(
        scene, veh_idx, roadway, R, F, R)
    rear_R = get_neighbor_rear_along_right_lane(
        scene, veh_idx, roadway, R, F, R)
    if fore_M.ind != 0
        fore_fore_M = get_neighbor_fore_along_lane(
            scene, fore_M.ind, roadway, F, R, F)
    else
        fore_fore_M = NeighborLongitudinalResult(0, 0.)
    end

    next_feature_idx = 28
    set_neighbor_features!(
        features, next_feature_idx, fore_M, scene, rec, roadway, veh_idx)
    set_neighbor_features!(
        features, next_feature_idx += 5, fore_L, scene, rec, roadway, veh_idx)
    set_neighbor_features!(
        features, next_feature_idx += 5, fore_R, scene, rec, roadway, veh_idx)
    set_neighbor_features!(
        features, next_feature_idx += 5, rear_M, scene, rec, roadway, veh_idx)
    set_neighbor_features!(
        features, next_feature_idx += 5, rear_L, scene, rec, roadway, veh_idx)
    set_neighbor_features!(
        features, next_feature_idx += 5, rear_R, scene, rec, roadway, veh_idx)
    set_neighbor_features!(
        features, next_feature_idx += 5, fore_fore_M, scene, rec, roadway, veh_idx)

    # extract driver behavior features for ego and surrounding vehicles
    idxs::Vector{Int64} = [veh_idx, fore_M.ind, fore_L.ind, fore_R.ind, 
        rear_M.ind, rear_L.ind, rear_R.ind, fore_fore_M.ind]
    next_feature_idx = 62
    for veh_index in idxs
        next_feature_idx = set_behavioral_features(scene, models, features, 
            veh_index, next_feature_idx)
    end 
end