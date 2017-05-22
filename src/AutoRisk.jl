__precompile__(true)
module AutoRisk

using Reexport

@reexport using AutomotiveDrivingModels
@reexport using BayesNets
@reexport using DataStructures
@reexport using Discretizers
@reexport using Distributions
@reexport using ForwardNets
@reexport using HDF5
@reexport using Parameters
@reexport using GridInterpolations

import AutomotiveDrivingModels: 
    simulate!, update!, observe!, pull_features!, set_desired_speed!, observe!,
    get_name, show
import Base: display, show, rand, rand!, ==
import Distributions: rand, pdf, logpdf
import Discretizers: encode, decode

# utils
include("evaluation/simulation.jl")
include("utils/automotive.jl")
include("utils/utils.jl")
include("utils/assignment_sampler.jl")

# # behaviors
include("behaviors/errorable_driver_model.jl")
# include("behaviors/delayed_intelligent_driver_model.jl")
include("behaviors/delayed_driver_model.jl")
include("behaviors/gaussian_mlp_driver.jl")

# extraction
include("extraction/lidar_sensors.jl")
include("extraction/feature_extractors.jl")
include("extraction/multi_feature_extractor.jl")
include("extraction/dataset_extraction.jl")

# generation
## roadway
include("generation/roadway/roadway_generator.jl")

## scene
include("generation/scene/scene_generator.jl")
include("generation/scene/heuristic_scene_generator.jl")
include("generation/scene/dataset_scene_generator.jl")

# ## behavior
include("generation/behavior/parameters.jl")
include("generation/behavior/behavior_generator.jl")
include("generation/behavior/heuristic_behavior_generators.jl")
include("generation/behavior/load_policy.jl")
include("generation/behavior/learned_behavior_generators.jl")

# ## additional generation 
include("generation/generator.jl")
include("generation/bayes_net_lane_generator.jl")
include("generation/debug_generator.jl")

# prediction
include("prediction/neural_network.jl")
include("prediction/td_predictor.jl")

# evaluation
include("evaluation/monte_carlo_evaluator.jl")
include("evaluation/bootstrapping_monte_carlo_evaluator.jl")

# collection
include("collection/dataset.jl")
include("collection/dataset_collector.jl")

# Display portion of AutoRisk may be unnecessary or unavailable in some 
# environments, so optionally include that here if possible
# try
#     @reexport using AutoViz
#     @reexport using Interact
#     @reexport using Reel
#     # analysis
#     include("analysis/display.jl")
#     # monitoring is entirely visual for now
#     include("analysis/monitor.jl")
# catch e
#     println("Exception encountered in AutoRisk while trying to import display
#         libraries and functionality: $(e)")
# end

end # module