export 
    set_feature_missing!,
    set_feature!,
    set_dual_feature!,
    set_neighbor_features!,
    set_behavioral_features

function set_feature_missing!(features::Array{Float64}, i::Int, veh_idx::Int64)
    features[i, veh_idx] = 0.0
    features[i+1, veh_idx] = 1.0
end

function set_feature!(features::Array{Float64}, i::Int, v::Float64, 
        veh_idx::Int64)
    features[i, veh_idx] = v
    features[i+1, veh_idx] = 0.0
end

function set_dual_feature!(features::Array{Float64}, i::Int, 
        f::FeatureValue, veh_idx::Int64)
    if f.i == FeatureState.MISSING
        set_feature_missing!(features, i, veh_idx)
    else
        set_feature!(features, i, f.v, veh_idx)
    end
end

function set_neighbor_features!(features::Array{Float64}, i::Int, 
        neigh::NeighborLongitudinalResult, scene::Scene, rec::SceneRecord, 
        roadway::Roadway, veh_idx::Int64)
    if neigh.ind != 0
        features[i, veh_idx] = neigh.Δs
        features[i+1, veh_idx] = scene[neigh.ind].state.v
        features[i+2, veh_idx] = convert(Float64, get(ACC, rec, roadway, neigh.ind))
        features[i+3, veh_idx] = convert(Float64, get(JERK, rec, roadway, neigh.ind))
        features[i+4, veh_idx] = 0.0
    else
        features[i:i+3, veh_idx] = 0.0
        features[i+4, veh_idx] = 1.0 
    end
end

function set_behavioral_features(scene::Scene, models::Dict{Int, DriverModel},
        features::Array{Float64}, veh_idx::Int64, next_idx::Int64)
    # number of behavioral features set
    num_behavior_features = 13
    
    # if vehicle does not exist then leave features as zeros
    if veh_idx == 0
        return next_idx + num_behavior_features
    end

    # get the vehicle model
    veh = scene[veh_idx]
    model = models[veh.def.id]
    if typeof(model) == DelayedDriver
        mlon = model.driver.mlon
        mlat = model.driver.mlat
        mlane = model.driver.mlane
    else
        mlon = model.mlon
        mlat = model.mlat
        mlane = model.mlane
    end

    # set behavioral features
    # longitudinal model
    if (typeof(mlon) == IntelligentDriverModel 
            || typeof(mlon) == DelayedIntelligentDriverModel)
        features[next_idx, veh_idx] = mlon.k_spd
        features[next_idx += 1, veh_idx] = mlon.δ
        features[next_idx += 1, veh_idx] = mlon.T
        features[next_idx += 1, veh_idx] = mlon.v_des
        features[next_idx += 1, veh_idx] = mlon.s_min
        features[next_idx += 1, veh_idx] = mlon.a_max
        features[next_idx += 1, veh_idx] = mlon.d_cmf
        if typeof(mlon) == DelayedIntelligentDriverModel
            features[next_idx += 1, veh_idx] = mlon.t_d
        elseif typeof(model) == DelayedDriver
            features[next_idx += 1, veh_idx] = model.reaction_time
        else
            next_idx += 1
        end
    else
        # one less then the number of features since the first doesn't increment
        next_idx += 7
    end

    # lateral model
    if typeof(mlat) == ProportionalLaneTracker
        features[next_idx += 1, veh_idx] = mlat.kp
        features[next_idx += 1, veh_idx] = mlat.kd
    else
        next_idx += 2
    end

    # lane model
    if typeof(mlane) == MOBIL
        features[next_idx += 1, veh_idx] = mlane.politeness
        features[next_idx += 1, veh_idx] = mlane.advantage_threshold
        features[next_idx += 1, veh_idx] = mlane.safe_decel
    else
        next_idx += 3
    end

    # return next_idx + 1 to know where to start setting feature values
    return next_idx + 1
end