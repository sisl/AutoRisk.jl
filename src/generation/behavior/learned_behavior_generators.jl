
export 
    LearnedBehaviorGenerator,
    reset!,
    rand

type LearnedBehaviorGenerator <: BehaviorGenerator
    model::DriverModel
    function LearnedBehaviorGenerator(filepath::String)
        extractor = MultiFeatureExtractor(filepath)
        gru_layer = contains(filepath, "gru")
        model = load_gaussian_mlp_driver(filepath, extractor, 
            gru_layer = gru_layer)
        return new(model)
    end
end

function reset!(gen::LearnedBehaviorGenerator, models::Dict{Int, DriverModel}, 
        scene::Scene, seed::Int64)
    # only populate with the single model 
    if length(models) == 0
        for veh in scene.vehicles
            models[veh.def.id] = gen.model
        end
    end
    return models
    # only populate with the single model 
    # if length(models) == 0
    #     for (i, veh) in enumerate(scene.vehicles)
    #         if i % 10 == 0
    #             models[veh.def.id] = gen.model
    #         else
    #             models[veh.def.id] = Tim2DDriver(IntegratedContinuous(.1, 1))
    #         end
    #     end
    # end
    # return models
end