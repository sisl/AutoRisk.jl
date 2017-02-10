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