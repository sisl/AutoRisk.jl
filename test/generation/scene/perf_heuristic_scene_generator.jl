using Base.Test
using AutoRisk

function build_debug_heuristic_scene_generator(;
        min_num_vehicles = 4, 
        max_num_vehicles = 100, 
        min_base_speed = 30.,
        max_base_speed = 30.,
        min_vehicle_length = 3.,
        max_vehicle_length = 7.,
        min_vehicle_width = 1.0, 
        max_vehicle_width = 3.0,
        min_init_dist = 10., 
        max_init_dist = 30.,
        rng = MersenneTwister(1))
    return HeuristicSceneGenerator(
        min_num_vehicles, 
        max_num_vehicles, 
        min_base_speed,
        max_base_speed,
        min_vehicle_length,
        max_vehicle_length,
        min_vehicle_width, 
        max_vehicle_width,
        min_init_dist, 
        max_init_dist,
        rng)
end

function perf_generate_init_road_idxs(;num_runs = 1)
    num_vehicles = 300
    gen = build_debug_heuristic_scene_generator(
        max_num_vehicles = num_vehicles)
    gen.total_roadway_length = 200. + 50. * pi * 2 
    roadway = gen_stadium_roadway(1, length = 400., radius = 100.)
    for run in 1:num_runs
        init = generate_init_road_idxs(gen, roadway, num_vehicles)
    end
    
end

perf_generate_init_road_idxs()
@time perf_generate_init_road_idxs(num_runs = 1000)
