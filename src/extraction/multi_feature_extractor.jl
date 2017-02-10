export 
    MultiFeatureExtractor,
    length,
    pull_features!

type MultiFeatureExtractor <: AbstractFeatureExtractor
    extractors::Vector{AbstractFeatureExtractor}
end

function Base.length(ext::MultiFeatureExtractor)
    return sum(length(subext) for subext in ext.extractors)
end
function AutomotiveDrivingModels.pull_features!(
        ext::MultiFeatureExtractor, 
        features::Array{Float64}, 
        rec::SceneRecord,
        roadway::Roadway, 
        vehicle_index::Int,  
        models = Dict{Int, DriverModel},
        fidx::Int = 0,
        pastframe::Int = 0)
    # each sub extractor is passed the index prior to where it should begin 
    # inserting values
    feature_index = 0
    for subext in ext.extractors
        pull_features!(subext, features, rec, roadway, vehicle_index, models, feature_index, pastframe)
        feature_index += length(subext)
    end
    return features
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