__precompile__(true)
module AutoRisk

using Reexport

@reexport using AutomotiveDrivingModels
@reexport using DataStructures
@reexport using Distributions
@reexport using HDF5

import AutomotiveDrivingModels: simulate!, update!, reset!, observe!
import Base: display, show, rand, ==
import Distributions: rand, pdf, logpdf

# utils
include("utils/automotive.jl")
include("utils/flags.jl")
include("utils/utils.jl")

# extraction
include("extraction/lidar_sensors.jl")
include("extraction/feature_extractors.jl")
include("extraction/heuristic_feature_extractor.jl")
include("extraction/multi_feature_extractor.jl")

# generation
## roadway
include("generation/roadway/roadway_generator.jl")

## scene
include("generation/scene/scene_generator.jl")
include("generation/scene/heuristic_scene_generator.jl")
include("generation/scene/dataset_scene_generator.jl")

## behavior
include("generation/behavior/parameters.jl")
include("generation/behavior/behavior_generator.jl")
include("generation/behavior/heuristic_behavior_generators.jl")
include("generation/behavior/delayed_intelligent_driver_model.jl")
include("generation/behavior/delayed_driver_model.jl")

# prediction
include("prediction/neural_network.jl")

# evaluation
include("evaluation/simulation.jl")
include("evaluation/dataset_extraction.jl")
include("evaluation/monte_carlo_evaluator.jl")
include("evaluation/bootstrapping_monte_carlo_evaluator.jl")

# collection
include("collection/dataset.jl")
include("collection/dataset_collector.jl")


# Display portion of AutoRisk may be unnecessary or unavailable in some 
# environments, so optionally include that here if possible
try
    @reexport using AutoViz
    @reexport using Interact
    # analysis
    include("analysis/display.jl")
catch e
    println("Exception encountered in AutoRisk while trying to import display
        libraries and functionality: $(e)")
end

end # module