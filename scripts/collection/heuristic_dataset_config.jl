FLAGS = Flags()

# scene constants
# roadway
add_entry!(FLAGS, 
    "num_lanes", 5, Int64, 
    "Number of lanes in the simulations.")
add_entry!(FLAGS, 
    "roadway_length", 400., Float64, 
    "Length of the roadway.")
add_entry!(FLAGS, 
    "roadway_radius", 100., Float64, 
    "Radius of turns in the roadway.")
add_entry!(FLAGS, 
    "lane_width", 3., Float64, 
    "Width of lane in meters.")

# vehicles
add_entry!(FLAGS, 
    "min_num_vehicles", 200, Int64, 
    "Number of vehicles on the road.")
add_entry!(FLAGS, 
    "max_num_vehicles", 200, Int64, 
    "Number of vehicles on the road.")
add_entry!(FLAGS, 
    "min_base_speed", 8., Float64, 
    "Base vehicle speed.")
add_entry!(FLAGS, 
    "max_base_speed", 12., Float64, 
    "Base vehicle speed.")
add_entry!(FLAGS, 
    "min_vehicle_length", 3., Float64, 
    "Vehicle length.")
add_entry!(FLAGS, 
    "max_vehicle_length", 6., Float64, 
    "Vehicle length.")
add_entry!(FLAGS, 
    "min_vehicle_width", 1.2, Float64, 
    "Vehicle width.")
add_entry!(FLAGS, 
    "max_vehicle_width", 2.6, Float64, 
    "Vehicle width.")
add_entry!(FLAGS, 
    "min_init_dist", 12., Float64, 
    "Minimum distance between vehicles at start of simulation.")

# behavior
add_entry!(FLAGS, 
    "behavior_type", "", String, 
    "Only use this behavior {aggressive, passive, normal} if given.")
add_entry!(FLAGS, 
    "behavior_noise", true, Bool, 
    "Whether or not driving behaviors should have noise applied to actions.")
add_entry!(FLAGS, 
    "delayed_response", true, Bool, 
    "Whether or not longitudinal acceleration is delayed.")

# simulation constants
add_entry!(FLAGS, 
    "num_scenarios", 100, Int64, 
    "Number of unique trajectories.")
add_entry!(FLAGS, 
    "num_monte_carlo_runs", 40, Int64, 
    "Number of monte carlo runs per trajectory.")
add_entry!(FLAGS, 
    "sampling_period", .1, Float64, 
    "Number of samples per second (i.e., hz).")
add_entry!(FLAGS, 
    "prime_time", 10., Float64, 
    "Time before collecting target values.")
add_entry!(FLAGS, 
    "sampling_time", 10., Float64, 
    "Seconds to simulate trajectory.")
add_entry!(FLAGS, 
    "num_proc", 1, Int64, 
    "Number of parallel processes for gathering dataset.")
add_entry!(FLAGS, 
    "verbose", 1, Int64, 
    "Level of verbosity in outputting information.")

# NGSIM constants
add_entry!(FLAGS, 
    "num_frames_threshold", 80, Int64, 
    "Number of frames a vehicle must be in to be sampled (10 is 1 sec).")
add_entry!(FLAGS, 
    "NGSIM_i101_set", 1, Int64, 
    "Which NGSIM time period to use {1,2,3}.")
add_entry!(FLAGS, 
    "NGSIM_prime_time", 1., Float64, 
    "Time before collecting target values.")
add_entry!(FLAGS, 
    "NGSIM_max_num_vehicles", 193, Int64, 
    "Number of vehicles on the road.")
add_entry!(FLAGS, 
    "NGSIM_num_resets", 1000, Int64, 
    "Number of vehicles on the road.")

# evaluator constants
add_entry!(FLAGS, 
    "evaluator_type", "base", String, 
    "Type of evaluator to use {base, bootstrap}.")
add_entry!(FLAGS, 
    "prediction_model_type", "neural_network", String, 
    "Type of prediction model to use {neural_network}.")
add_entry!(FLAGS, 
    "network_filepath", "../../data/networks/network.weights", String, 
    "Filepath to network weights file, generally for bootstrapping.")
## extractor 
add_entry!(FLAGS, 
    "extractor_type", "heuristic", String, 
    "Type of extractor to use {heuristic, multi}.")
add_entry!(FLAGS, 
    "extract_core", true, Bool, 
    "When using multi extractor, extract core features.")
add_entry!(FLAGS, 
    "extract_temporal", true, Bool, 
    "When using multi extractor, extract temporal features.")
add_entry!(FLAGS, 
    "extract_well_behaved", true, Bool, 
    "When using multi extractor, extract well behaved features.")
add_entry!(FLAGS, 
    "extract_neighbor", true, Bool, 
    "When using multi extractor, extract neighbor features.")
add_entry!(FLAGS, 
    "extract_car_lidar", true, Bool, 
    "When using multi extractor, extract car lidar features.")
add_entry!(FLAGS, 
    "extract_car_lidar_range_rate", true, Bool, 
    "When using multi extractor, extract car lidar range rate features.")
add_entry!(FLAGS, 
    "extract_road_lidar", true, Bool, 
    "When using multi extractor, extract road lidar features.")

# dataset constants
add_entry!(FLAGS, 
    "dataset_type", "heuristic", String, 
    "Type of dataset to generate.")
add_entry!(FLAGS, 
    "target_dim", 5, Int64, 
    "Number of target values (e.g., p(collision).")
add_entry!(FLAGS, 
    "chunk_dim", 10, Int64, 
    "Dimension of dataset chunking.")
add_entry!(FLAGS, 
    "output_filepath", "../../data/datasets/risk.h5", String, 
    "Filepath where to save dataset.")
add_entry!(FLAGS, 
    "initial_seed", 1, Int64, 
    "If using a sequential set of seed values, this is the first that will be used.")