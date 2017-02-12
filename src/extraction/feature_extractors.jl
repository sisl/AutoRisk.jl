export 
    set_feature_missing!,
    set_feature!,
    set_dual_feature!,
    set_neighbor_features!,
    set_behavioral_features,
    CoreFeatureExtractor,
    TemporalFeatureExtractor,
    WellBehavedFeatureExtractor,
    NeighborFeatureExtractor,
    CarLidarFeatureExtractor,
    RoadLidarFeatureExtractor,
    NormalizingExtractor,
    EmptyExtractor,
    pull_features!,
    length

##################### Helper methods #####################

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

function set_speed_and_distance!(features::Array{Float64}, i::Int, 
    neigh::NeighborLongitudinalResult, scene::Scene, veh_idx::Int64)
    neigh.ind != 0 ? set_feature!(features, i, scene[neigh.ind].state.v, veh_idx) :
                      set_feature_missing!(features, i, veh_idx)
    neigh.ind != 0 ? set_feature!(features, i+2, neigh.Δs, veh_idx) :
                      set_feature_missing!(features, i+2, veh_idx)
    features
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

##################### Specific Feature Extractors #####################

type CoreFeatureExtractor <: AbstractFeatureExtractor
end
Base.length(ext::CoreFeatureExtractor) = 8
function AutomotiveDrivingModels.pull_features!(
        ext::CoreFeatureExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,  
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        fidx::Int = 0,
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)
    veh_ego = scene[veh_idx]
    d_ml = convert(Float64, get(
        MARKERDIST_LEFT, rec, roadway, veh_idx, pastframe))
    d_mr = convert(Float64, get(
        MARKERDIST_RIGHT, rec, roadway, veh_idx, pastframe))

    features[fidx+=1, veh_idx] = veh_ego.state.posF.t
    features[fidx+=1, veh_idx] = veh_ego.state.posF.ϕ
    features[fidx+=1, veh_idx] = veh_ego.state.v
    features[fidx+=1, veh_idx] = veh_ego.def.length
    features[fidx+=1, veh_idx] = veh_ego.def.width
    features[fidx+=1, veh_idx] = convert(Float64, get(
        LANECURVATURE, rec, roadway, veh_idx, pastframe))
    features[fidx+=1, veh_idx] = d_ml
    features[fidx+=1, veh_idx] = d_mr
end
    
type TemporalFeatureExtractor <: AbstractFeatureExtractor
end
Base.length(ext::TemporalFeatureExtractor) = 6
function AutomotiveDrivingModels.pull_features!(
        ext::TemporalFeatureExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int, 
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        fidx::Int = 0,
        pastframe::Int = 0)
    features[fidx+=1, veh_idx] = convert(Float64, get(
        ACC, rec, roadway, veh_idx, pastframe))
    features[fidx+=1, veh_idx] = convert(Float64, get(
        JERK, rec, roadway, veh_idx, pastframe))
    features[fidx+=1, veh_idx] = convert(Float64, get(
        TURNRATEG, rec, roadway, veh_idx, pastframe))
    features[fidx+=1, veh_idx] = convert(Float64, get(
        ANGULARRATEG, rec, roadway, veh_idx, pastframe))
    features[fidx+=1, veh_idx] = convert(Float64, get(
        TURNRATEF, rec, roadway, veh_idx, pastframe))
    features[fidx+=1, veh_idx] = convert(Float64, get(
        ANGULARRATEF, rec, roadway, veh_idx, pastframe))
end

type WellBehavedFeatureExtractor <: AbstractFeatureExtractor
end
Base.length(ext::WellBehavedFeatureExtractor) = 3
function AutomotiveDrivingModels.pull_features!(
        ext::WellBehavedFeatureExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int, 
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        fidx::Int = 0,
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)
    veh_ego = scene[veh_idx]
    d_ml = convert(Float64, get(
        MARKERDIST_LEFT, rec, roadway, veh_idx, pastframe))
    d_mr = convert(Float64, get(
        MARKERDIST_RIGHT, rec, roadway, veh_idx, pastframe))
    features[fidx+=1, veh_idx] = convert(Float64, get(
        IS_COLLIDING, rec, roadway, veh_idx, pastframe))
    features[fidx+=1, veh_idx] = convert(Float64, d_ml < -1.0 || d_mr < -1.0)
    features[fidx+=1, veh_idx] = convert(Float64, veh_ego.state.v < 0.0)
end

type NeighborFeatureExtractor <: AbstractFeatureExtractor
end
Base.length(ext::NeighborFeatureExtractor) = 28
function AutomotiveDrivingModels.pull_features!(
        ext::NeighborFeatureExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int, 
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        fidx::Int = 0,
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)

    vtpf = VehicleTargetPointFront()
    vtpr = VehicleTargetPointRear()
    fore_M = get_neighbor_fore_along_lane(
        scene, veh_idx, roadway, vtpf, vtpr, vtpf)
    fore_L = get_neighbor_fore_along_left_lane(
        scene, veh_idx, roadway, vtpf, vtpr, vtpf)
    fore_R = get_neighbor_fore_along_right_lane(
        scene, veh_idx, roadway, vtpf, vtpr, vtpf)
    rear_M = get_neighbor_rear_along_lane(
        scene, veh_idx, roadway, vtpr, vtpf, vtpr)
    rear_L = get_neighbor_rear_along_left_lane(
        scene, veh_idx, roadway, vtpr, vtpf, vtpr)
    rear_R = get_neighbor_rear_along_right_lane(
        scene, veh_idx, roadway, vtpr, vtpf, vtpr)

    set_dual_feature!(features, fidx+=1, get(
        LANEOFFSETLEFT, rec, roadway, veh_idx, pastframe), veh_idx)
    fidx+=1
    set_dual_feature!(features, fidx+=1, get(
        LANEOFFSETRIGHT, rec, roadway, veh_idx, pastframe), veh_idx)
    fidx+=1

    set_speed_and_distance!(features, fidx+=1, fore_M, scene, veh_idx)
    fidx+=3
    set_speed_and_distance!(features, fidx+=1, fore_L, scene, veh_idx)
    fidx+=3
    set_speed_and_distance!(features, fidx+=1, fore_R, scene, veh_idx)
    fidx+=3
    set_speed_and_distance!(features, fidx+=1, rear_M, scene, veh_idx)
    fidx+=3
    set_speed_and_distance!(features, fidx+=1, rear_L, scene, veh_idx)
    fidx+=3
    set_speed_and_distance!(features, fidx+=1, rear_R, scene, veh_idx)
    fidx+=3
end

type CarLidarFeatureExtractor <: AbstractFeatureExtractor
    carlidar::LidarSensor
    extract_carlidar_rangerate::Bool
    function CarLidarFeatureExtractor(
            carlidar_nbeams::Int = 20; 
            extract_carlidar_rangerate::Bool = true,
            carlidar_max_range::Float64 = 100.0)
        carlidar = LidarSensor(carlidar_nbeams, max_range=carlidar_max_range, angle_offset=-π)
        new(carlidar, extract_carlidar_rangerate)
    end
end
Base.length(ext::CarLidarFeatureExtractor) = nbeams(ext.carlidar) * (
    1 + ext.extract_carlidar_rangerate)
function AutomotiveDrivingModels.pull_features!(
        ext::CarLidarFeatureExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int, 
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        fidx::Int = 0,
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)
    nbeams_carlidar = nbeams(ext.carlidar)
    if nbeams_carlidar > 0
        observe!(ext.carlidar, scene, roadway, veh_idx)
        stop = length(ext.carlidar.ranges) + fidx
        fidx += 1
        features[fidx:stop, veh_idx] = ext.carlidar.ranges
        fidx += nbeams_carlidar - 1
        if ext.extract_carlidar_rangerate
            stop = length(ext.carlidar.range_rates) + fidx
            fidx += 1
            features[fidx:stop, veh_idx] = ext.carlidar.range_rates
            fidx += nbeams_carlidar - 1
        end
    end
end

type RoadLidarFeatureExtractor <: AbstractFeatureExtractor
    roadlidar::RoadlineLidarSensor
    road_lidar_culling::RoadwayLidarCulling
    function RoadLidarFeatureExtractor(
            roadlidar_nbeams::Int = 20,
            roadlidar_nlanes::Int = 2,
            roadlidar_max_range::Float64 = 100.0)
        roadlidar = RoadlineLidarSensor(roadlidar_nbeams, max_range=roadlidar_max_range, angle_offset=-π, max_depth=roadlidar_nlanes)
        return new(roadlidar, RoadwayLidarCulling())
    end
end
Base.length(ext::RoadLidarFeatureExtractor) = nbeams(ext.roadlidar) * nlanes(
    ext.roadlidar)
function AutomotiveDrivingModels.pull_features!(
        ext::RoadLidarFeatureExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,  
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        fidx::Int = 0,
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)
    nbeams_roadlidar = nbeams(ext.roadlidar)
    if nbeams_roadlidar > 0
        if ext.road_lidar_culling.is_leaf
            observe!(ext.roadlidar, scene, roadway, veh_idx)
        else
            observe!(ext.roadlidar, scene, roadway, veh_idx, ext.road_lidar_culling)
        end
        
        stop = length(ext.roadlidar.ranges) + fidx
        fidx += 1
        features[fidx:stop, veh_idx] = reshape(ext.roadlidar.ranges, 
            length(ext.roadlidar.ranges))
        fidx += length(ext.roadlidar.ranges) - 1
    end
end

##################### Feature Extractor Wrappers #####################

type NormalizingExtractor <: AbstractFeatureExtractor
    μ::Vector{Float64}
    σ::Vector{Float64}
    extractor::AbstractFeatureExtractor
end
Base.length(ext::NormalizingExtractor) = length(ext.extractor)
function AutomotiveDrivingModels.pull_features!(
        ext::NormalizingExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        fidx::Int = 0,
        pastframe::Int = 0)

    # extract base feature values
    pull_features!(ext.extractor, features, rec, roadway, veh_idx, models, 
        fidx, pastframe)

    # normalize those values
    features[:] = (features .- ext.μ) ./ ext.σ
    return features
end

type EmptyExtractor <: AbstractFeatureExtractor
end
Base.length(ext::EmptyExtractor) = 0
function AutomotiveDrivingModels.pull_features!(
        ext::EmptyExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        fidx::Int = 0,
        pastframe::Int = 0)
    return features
end
