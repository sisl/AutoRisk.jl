using Base.Test
using AutoRisk

include("collection/testing_utils.jl")

# testing constants
const NUM_FEATURES = 276
const NUM_TARGETS = 5
const BASE_TEST_DIR = "."

function runtests()
    # utils
    println("\n### utils ###")
    println("test_utils.jl")
    include("utils/test_utils.jl")
    println("test_automotive.jl")
    include("utils/test_automotive.jl")
    println("test_assignment_sampler.jl")
    include("utils/test_assignment_sampler.jl")

    # extraction
    println("\n### extraction ###")
    println("test_feature_extractors.jl")
    include("extraction/test_feature_extractors.jl")
    println("test_multi_feature_extractor.jl")
    include("extraction/test_multi_feature_extractor.jl")
    println("test_dataset_extraction.jl")
    include("extraction/test_dataset_extraction.jl")

    # behavior
    println("\n### behaviors ###")
    println("test_delayed_intelligent_driver_model.jl")
    include("behaviors/test_delayed_intelligent_driver_model.jl")
    println("test_delayed_driver_model.jl")
    include("behaviors/test_delayed_driver_model.jl")
    println("test_errorable_driver_model.jl")
    include("behaviors/test_errorable_driver_model.jl")

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
    println("test_debug_generator.jl")
    include("generation/test_debug_generator.jl")
    ## combined 
    println("test_bayes_net_lane_generator.jl")
    include("generation/test_bayes_net_lane_generator.jl")

    # evaluation
    println("\n### evaluation ###")
    println("test_simulation.jl")
    include("evaluation/test_simulation.jl")
    println("test_monte_carlo_evaluator.jl")
    include("evaluation/test_monte_carlo_evaluator.jl")
    println("test_bootstrapping_monte_carlo_evaluator.jl")
    include("evaluation/test_bootstrapping_monte_carlo_evaluator.jl")

    # prediction
    println("\n### prediction ###")
    println("test_td_predictor.jl")
    include("prediction/test_td_predictor.jl")

    # collection
    println("\n### collection ###")
    println("test_dataset.jl")
    include("collection/test_dataset.jl")
    println("test_dataset_collector.jl")
    include("collection/test_dataset_collector.jl")
    println("test_parallel_dataset_collector.jl")
    include("collection/test_parallel_dataset_collector.jl")

    println("\nAll tests pass!")
end

@time runtests()
