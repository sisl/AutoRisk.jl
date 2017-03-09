export 
    MultiFeatureExtractor,
    length,
    pull_features!,
    feature_names

type MultiFeatureExtractor <: AbstractFeatureExtractor
    extractors::Vector{AbstractFeatureExtractor}
    lengths::Vector{Int64}
    features::Vector{Float64}
    num_features::Int64
    function MultiFeatureExtractor(extractors::Vector{AbstractFeatureExtractor})
        lengths = [length(subext) for subext in extractors]
        num_features = sum(lengths)
        features = zeros(Float64, num_features)
        new(extractors, lengths, features, num_features)
    end
end
Base.length(ext::MultiFeatureExtractor) = ext.num_features
function feature_names(ext::MultiFeatureExtractor)
    fs = []
    for subext in ext.extractors
        push!(fs, feature_names(subext))
    end
    return cat(1, fs...)
end
function AutomotiveDrivingModels.pull_features!(
        ext::MultiFeatureExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        vehicle_index::Int,  
        models = Dict{Int, DriverModel},
        pastframe::Int = 0)
    feature_index = 1
    for (subext, len) in zip(ext.extractors, ext.lengths)
        stop = feature_index + len - 1
        ext.features[feature_index:stop] = pull_features!(
            subext, rec, roadway, vehicle_index, models, pastframe)
        feature_index += len
    end
    return ext.features
end

# alternate constructor containing all sub-extractors
function MultiFeatureExtractor()
    subexts = [
        CoreFeatureExtractor(),
        TemporalFeatureExtractor(),
        WellBehavedFeatureExtractor(),
        NeighborFeatureExtractor(),
        BehavioralFeatureExtractor(),
        NeighborBehavioralFeatureExtractor(),
        CarLidarFeatureExtractor(),
        RoadLidarFeatureExtractor()
    ]

   return MultiFeatureExtractor(subexts)
end

# alternate constructor from h5 file with attributes
function MultiFeatureExtractor(filepath::String)
    # open the file
    file = h5open(filepath, "r")

    # get feature set information from file attributes
    extract_core = a_read(file, "extract_core")
    extract_temporal = a_read(file, "extract_temporal")
    extract_well_behaved = a_read(file, "extract_well_behaved")
    extract_neighbor = a_read(file, "extract_neighbor")
    extract_car_lidar = a_read(file, "extract_car_lidar")
    extract_car_lidar_range_rate = a_read(file, "extract_car_lidar_range_rate")
    extract_road_lidar = a_read(file, "extract_road_lidar")

    # close file after gathering attributes
    close(file)

    # create the sub extractors
    subexts = []
    if Bool(extract_core) == true
        push!(subexts, CoreFeatureExtractor())
    end
    if Bool(extract_temporal) == true
        push!(subexts, TemporalFeatureExtractor())
    end
    if Bool(extract_well_behaved) == true
        push!(subexts, WellBehavedFeatureExtractor())
    end
    if Bool(extract_neighbor) == true
        push!(subexts, NeighborFeatureExtractor())
    end
    if Bool(extract_car_lidar) == true
        push!(subexts, 
            CarLidarFeatureExtractor(
                extract_carlidar_rangerate = 
                Bool(extract_car_lidar_range_rate)))
    end
    if Bool(extract_road_lidar) == true
        push!(subexts, RoadLidarFeatureExtractor())
    end
    
    # build the multi extractor
    return MultiFeatureExtractor(subexts)
end