__precompile__()
module AutoRisk

using Reexport

@reexport using AutomotiveDrivingModels
@reexport using AutoViz
@reexport using DataStructures
@reexport using Distributions
@reexport using HDF5
@reexport using Interact

import AutomotiveDrivingModels: simulate!, update!, reset!
import Base: display, show, rand, ==
import Distributions: rand, pdf, logpdf

# utils
include("utils/automotive.jl")
include("utils/flags.jl")
include("utils/utils.jl")

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

# analysis
include("analysis/display.jl")

end