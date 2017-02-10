using Base.Test
using AutoRisk

# testing constants
const NUM_FEATURES = 166
const NUM_TARGETS = 5

function runtests()
    # utils
    println("\n### utils ###")
    println("test_utils.jl")
    include("utils/test_utils.jl")
    println("test_automotive.jl")
    include("utils/test_automotive.jl")

    # extraction
    println("\n### extraction ###")
    println("test_heuristic_feature_extractor.jl")
    include("extraction/test_heuristic_feature_extractor.jl")

    # generation
    println("\n### generation ###")
    ## scene
    println("test_heuristic_scene_generator.jl")
    include("generation/scene/test_heuristic_scene_generator.jl")
    ## behavior
    println("test_behavior_generator.jl")
    include("generation/behavior/test_behavior_generator.jl")
    println("test_heuristic_behavior_generators.jl")
    include("generation/behavior/test_heuristic_behavior_generators.jl")
    println("test_delayed_intelligent_driver_model.jl")
    include("generation/behavior/test_delayed_intelligent_driver_model.jl")
    println("test_delayed_driver_model.jl")
    include("generation/behavior/test_delayed_driver_model.jl")

    # evaluation
    println("\n### evaluation ###")
    println("test_simulation.jl")
    include("evaluation/test_simulation.jl")
    println("test_dataset_extraction.jl")
    include("evaluation/test_dataset_extraction.jl")
    println("test_monte_carlo_evaluator.jl")
    include("evaluation/test_monte_carlo_evaluator.jl")
    println("test_bootstrapping_monte_carlo_evaluator.jl")
    include("evaluation/test_bootstrapping_monte_carlo_evaluator.jl")

    # collection
    println("\n### collection ###")
    include("collection/testing_utils.jl")
    println("test_dataset.jl")
    include("collection/test_dataset.jl")
    println("test_dataset_collector.jl")
    include("collection/test_dataset_collector.jl")
    println("test_parallel_dataset_collector.jl")
    include("collection/test_parallel_dataset_collector.jl")

    println("\nAll tests pass!")
end

@time runtests()
