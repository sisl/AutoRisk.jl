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
    BehavioralFeatureExtractor,
    NeighborBehavioralFeatureExtractor,
    CarLidarFeatureExtractor,
    RoadLidarFeatureExtractor,
    NormalizingExtractor,
    EmptyExtractor,
    pull_features!,
    length

##################### Helper methods #####################

function set_feature_missing!(features::Vector{Float64}, i::Int)
    features[i] = 0.0
    features[i+1] = 1.0
end

function set_feature!(features::Vector{Float64}, i::Int, v::Float64)
    features[i] = v
    features[i+1] = 0.0
end

function set_dual_feature!(features::Vector{Float64}, i::Int, 
        f::FeatureValue)
    if f.i == FeatureState.MISSING
        set_feature_missing!(features, i)
    else
        set_feature!(features, i, f.v)
    end
end

function set_speed_and_distance!(features::Vector{Float64}, i::Int, 
    neigh::NeighborLongitudinalResult, scene::Scene)
    neigh.ind != 0 ? set_feature!(features, i, scene[neigh.ind].state.v) :
                      set_feature_missing!(features, i)
    neigh.ind != 0 ? set_feature!(features, i+2, neigh.Δs) :
                      set_feature_missing!(features, i+2)
    features
end

function set_neighbor_features!(features::Vector{Float64}, i::Int, 
        neigh::NeighborLongitudinalResult, scene::Scene, rec::SceneRecord, 
        roadway::Roadway)
    if neigh.ind != 0
        features[i] = neigh.Δs
        features[i+1] = scene[neigh.ind].state.v
        features[i+2] = convert(Float64, get(ACC, rec, roadway, neigh.ind))
        features[i+3] = convert(Float64, get(JERK, rec, roadway, neigh.ind))
        features[i+4] = 0.0
    else
        features[i:i+3] = 0.0
        features[i+4] = 1.0 
    end
end

##################### Specific Feature Extractors #####################

type CoreFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    function CoreFeatureExtractor()
        num_features = 8
        return new(zeros(Float64, num_features), num_features)
    end
end
Base.length(ext::CoreFeatureExtractor) = ext.num_features
function AutomotiveDrivingModels.pull_features!(
        ext::CoreFeatureExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,  
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)
    veh_ego = scene[veh_idx]
    d_ml = convert(Float64, get(
        MARKERDIST_LEFT, rec, roadway, veh_idx, pastframe))
    d_mr = convert(Float64, get(
        MARKERDIST_RIGHT, rec, roadway, veh_idx, pastframe))
    idx = 0
    ext.features[idx+=1] = veh_ego.state.posF.t
    ext.features[idx+=1] = veh_ego.state.posF.ϕ
    ext.features[idx+=1] = veh_ego.state.v
    ext.features[idx+=1] = veh_ego.def.length
    ext.features[idx+=1] = veh_ego.def.width
    ext.features[idx+=1] = convert(Float64, get(
        LANECURVATURE, rec, roadway, veh_idx, pastframe))
    ext.features[idx+=1] = d_ml
    ext.features[idx+=1] = d_mr
    return ext.features
end
    
type TemporalFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    function TemporalFeatureExtractor()
        num_features = 10
        return new(zeros(Float64, num_features), num_features)
    end
end
Base.length(ext::TemporalFeatureExtractor) = ext.num_features
function AutomotiveDrivingModels.pull_features!(
        ext::TemporalFeatureExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int, 
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)
    idx = 0
    ext.features[idx+=1] = convert(Float64, get(
        ACC, rec, roadway, veh_idx, pastframe))
    ext.features[idx+=1] = convert(Float64, get(
        JERK, rec, roadway, veh_idx, pastframe))
    ext.features[idx+=1] = convert(Float64, get(
        TURNRATEG, rec, roadway, veh_idx, pastframe))
    ext.features[idx+=1] = convert(Float64, get(
        ANGULARRATEG, rec, roadway, veh_idx, pastframe))
    ext.features[idx+=1] = convert(Float64, get(
        TURNRATEF, rec, roadway, veh_idx, pastframe))
    ext.features[idx+=1] = convert(Float64, get(
        ANGULARRATEF, rec, roadway, veh_idx, pastframe))

    # timegap is the time between when this vehicle's front bumper
    # will be in the position currently occupied by the vehicle 
    # infront's back bumper
    set_dual_feature!(ext.features, idx+=1, 
        get(TIMEGAP, rec, roadway, veh_idx, censor_hi = 30.0))
    idx+=1

    # inverse time to collision is the time until a collision 
    # assuming that no actions are taken
    # inverse is taken so as to avoid infinite value, so flip here to get back
    # to TTC
    inv_ttc = get(INV_TTC, rec, roadway, veh_idx)
    ttc = inverse_ttc_to_ttc(inv_ttc, censor_hi = 30.0)
    set_dual_feature!(ext.features, idx+=1, ttc)
    idx+=1
    return ext.features
end

type WellBehavedFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    function WellBehavedFeatureExtractor()
        num_features = 3
        return new(zeros(Float64, num_features), num_features)
    end
end
Base.length(ext::WellBehavedFeatureExtractor) = ext.num_features
function AutomotiveDrivingModels.pull_features!(
        ext::WellBehavedFeatureExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int, 
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)
    veh_ego = scene[veh_idx]
    d_ml = convert(Float64, get(
        MARKERDIST_LEFT, rec, roadway, veh_idx, pastframe))
    d_mr = convert(Float64, get(
        MARKERDIST_RIGHT, rec, roadway, veh_idx, pastframe))
    idx = 0
    ext.features[idx+=1] = convert(Float64, get(
        IS_COLLIDING, rec, roadway, veh_idx, pastframe))
    ext.features[idx+=1] = convert(Float64, d_ml < -1.0 || d_mr < -1.0)
    ext.features[idx+=1] = convert(Float64, veh_ego.state.v < 0.0)
    return ext.features
end

type NeighborFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    function NeighborFeatureExtractor()
        num_features = 28
        return new(zeros(Float64, num_features), num_features)
    end
end
Base.length(ext::NeighborFeatureExtractor) = ext.num_features
function AutomotiveDrivingModels.pull_features!(
        ext::NeighborFeatureExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int, 
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
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

    idx = 0
    set_dual_feature!(ext.features, idx+=1, get(
        LANEOFFSETLEFT, rec, roadway, veh_idx, pastframe))
    idx+=1
    set_dual_feature!(ext.features, idx+=1, get(
        LANEOFFSETRIGHT, rec, roadway, veh_idx, pastframe))
    idx+=1

    set_speed_and_distance!(ext.features, idx+=1, fore_M, scene)
    idx+=3
    set_speed_and_distance!(ext.features, idx+=1, fore_L, scene)
    idx+=3
    set_speed_and_distance!(ext.features, idx+=1, fore_R, scene)
    idx+=3
    set_speed_and_distance!(ext.features, idx+=1, rear_M, scene)
    idx+=3
    set_speed_and_distance!(ext.features, idx+=1, rear_L, scene)
    idx+=3
    set_speed_and_distance!(ext.features, idx+=1, rear_R, scene)
    idx+=3

    return ext.features
end

type BehavioralFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    function BehavioralFeatureExtractor()
        num_features = 13
        return new(zeros(Float64, num_features), num_features)
    end
end
Base.length(ext::BehavioralFeatureExtractor) = ext.num_features
function AutomotiveDrivingModels.pull_features!(
        ext::BehavioralFeatureExtractor,  
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,  
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)
    # if vehicle does not exist then leave features as zeros
    if veh_idx == 0
        return ext.features
    end

    # get the vehicle model
    scene = get_scene(rec, pastframe)
    veh = scene[veh_idx]
    model = models[veh.def.id]

    # unpack the underlying driver models
    # if the primary driver model is invalid for this extractor, then skip
    if typeof(model) == DelayedDriver
        mlon = model.driver.mlon
        mlat = model.driver.mlat
        mlane = model.driver.mlane
    elseif typeof(model) == Tim2DDriver
        mlon = model.mlon
        mlat = model.mlat
        mlane = model.mlane
    else # skip this extractor
        return ext.features
    end

    next_idx = 0

    # longitudinal model
    if (typeof(mlon) == IntelligentDriverModel 
            || typeof(mlon) == DelayedIntelligentDriverModel)
        ext.features[next_idx+=1] = mlon.k_spd
        ext.features[next_idx+=1] = mlon.δ
        ext.features[next_idx+=1] = mlon.T
        ext.features[next_idx+=1] = mlon.v_des
        ext.features[next_idx+=1] = mlon.s_min
        ext.features[next_idx+=1] = mlon.a_max
        ext.features[next_idx+=1] = mlon.d_cmf
        if typeof(mlon) == DelayedIntelligentDriverModel
            ext.features[next_idx+=1] = mlon.t_d
        elseif typeof(model) == DelayedDriver
            ext.features[next_idx+=1] = model.reaction_time
        else
            next_idx += 1
        end
    else
        next_idx += 8
    end

    # lateral model
    if typeof(mlat) == ProportionalLaneTracker
        ext.features[next_idx+=1] = mlat.kp
        ext.features[next_idx+=1] = mlat.kd
    else
        next_idx += 2
    end

    # lane model
    if typeof(mlane) == MOBIL
        ext.features[next_idx+=1] = mlane.politeness
        ext.features[next_idx+=1] = mlane.advantage_threshold
        ext.features[next_idx+=1] = mlane.safe_decel
    else
        next_idx += 3
    end

    return ext.features
end

type NeighborBehavioralFeatureExtractor <: AbstractFeatureExtractor
    subext::BehavioralFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    function NeighborBehavioralFeatureExtractor()
        subext = BehavioralFeatureExtractor()
        num_neighbors = 7
        num_features = length(subext) * num_neighbors
        return new(subext, zeros(Float64, num_features), num_features)
    end
end
Base.length(ext::NeighborBehavioralFeatureExtractor) = ext.num_features
function AutomotiveDrivingModels.pull_features!(
        ext::NeighborBehavioralFeatureExtractor,  
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,  
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
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

    if fore_M.ind != 0      
        fore_fore_M = get_neighbor_fore_along_lane(
            scene, fore_M.ind, roadway, vtpf, vtpr, vtpf)        
    else        
        fore_fore_M = NeighborLongitudinalResult(0, 0.)     
    end

    idxs::Vector{Int64} = [fore_M.ind, fore_L.ind, fore_R.ind, rear_M.ind, 
        rear_L.ind, rear_R.ind, fore_fore_M.ind]
    fidx = 0
    num_neigh_features = length(ext.subext)
    for neigh_veh_idx in idxs
        stop = fidx + num_neigh_features
        ext.features[fidx + 1:stop] = pull_features!(ext.subext, rec, roadway,
            neigh_veh_idx, models, pastframe)
        fidx += num_neigh_features
    end
    return ext.features
end


type CarLidarFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    carlidar::LidarSensor
    extract_carlidar_rangerate::Bool
    function CarLidarFeatureExtractor(
            carlidar_nbeams::Int = 20; 
            extract_carlidar_rangerate::Bool = true,
            carlidar_max_range::Float64 = 50.0)
        carlidar = LidarSensor(carlidar_nbeams, max_range=carlidar_max_range, angle_offset=-π)
        num_features = nbeams(carlidar) * (1 + extract_carlidar_rangerate)
        return new(zeros(Float64, num_features), num_features, carlidar,
            extract_carlidar_rangerate)
    end
end
Base.length(ext::CarLidarFeatureExtractor) = ext.num_features
function AutomotiveDrivingModels.pull_features!(
        ext::CarLidarFeatureExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int, 
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)
    nbeams_carlidar = nbeams(ext.carlidar)
    idx = 0
    if nbeams_carlidar > 0
        observe!(ext.carlidar, scene, roadway, veh_idx)
        stop = length(ext.carlidar.ranges) + idx
        idx += 1
        ext.features[idx:stop] = ext.carlidar.ranges
        idx += nbeams_carlidar - 1
        if ext.extract_carlidar_rangerate
            stop = length(ext.carlidar.range_rates) + idx
            idx += 1
            ext.features[idx:stop] = ext.carlidar.range_rates
            idx += nbeams_carlidar - 1
        end
    end
    return ext.features
end

type RoadLidarFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    roadlidar::RoadlineLidarSensor
    road_lidar_culling::RoadwayLidarCulling
    function RoadLidarFeatureExtractor(
            roadlidar_nbeams::Int = 20,
            roadlidar_nlanes::Int = 2,
            roadlidar_max_range::Float64 = 50.0)
        roadlidar = RoadlineLidarSensor(roadlidar_nbeams, 
            max_range=roadlidar_max_range, angle_offset=-π, 
            max_depth=roadlidar_nlanes)
        num_features = nbeams(roadlidar) * nlanes(roadlidar)
        return new(zeros(Float64, num_features), num_features, roadlidar,
            RoadwayLidarCulling())
    end
end
Base.length(ext::RoadLidarFeatureExtractor) = ext.num_features
function AutomotiveDrivingModels.pull_features!(
        ext::RoadLidarFeatureExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,  
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)
    scene = get_scene(rec, pastframe)
    nbeams_roadlidar = nbeams(ext.roadlidar)
    if nbeams_roadlidar > 0
        if ext.road_lidar_culling.is_leaf
            observe!(ext.roadlidar, scene, roadway, veh_idx)
        else
            observe!(ext.roadlidar, scene, roadway, veh_idx, ext.road_lidar_culling)
        end
        idx = 0
        stop = length(ext.roadlidar.ranges) + idx
        idx += 1
        ext.features[idx:stop] = reshape(ext.roadlidar.ranges, 
            length(ext.roadlidar.ranges))
        idx += length(ext.roadlidar.ranges) - 1
    end
    return ext.features
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
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)

    # extract base feature values
    return ((pull_features!(ext.extractor, rec, roadway, veh_idx, models, 
        pastframe) .- ext.μ) ./ ext.σ)
end

type EmptyExtractor <: AbstractFeatureExtractor
end
Base.length(ext::EmptyExtractor) = 0
function AutomotiveDrivingModels.pull_features!(
        ext::EmptyExtractor,  
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)
end
