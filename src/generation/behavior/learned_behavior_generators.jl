
export 
    LearnedBehaviorGenerator,
    rand!

type LearnedBehaviorGenerator <: BehaviorGenerator
    filepath::String
end
function Base.rand!(gen::LearnedBehaviorGenerator, models::Dict{Int, DriverModel}, 
        scene::Scene, seed::Int64)
    if length(models) == 0
        for veh in scene.vehicles
            if veh.def.id == 1
                extractor = MultiFeatureExtractor(gen.filepath)
                gru_layer = contains(gen.filepath, "gru")
                model = load_gaussian_mlp_driver(gen.filepath, extractor, 
                    gru_layer = gru_layer)
                models[veh.def.id] = model
            else
                models[veh.def.id] = Tim2DDriver(IntegratedContinuous(.1, 1))
            end
        end
    end
    return models
end