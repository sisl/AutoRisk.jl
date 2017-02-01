
using AutoRisk
using NGSIM

push!(LOAD_PATH, "../collection")

include("../collection/collect_heuristic_dataset.jl")
include("../collection/heuristic_dataset_config.jl")

function compute_density(scene::Scene, roadway::Roadway)
    return scene.n_vehicles / get_total_roadway_area(roadway)
end

function compute_heuristic_density(flags::Flags, seeds::Vector{Int})
    col = build_dataset_collector("", flags)
    total_density = 0
    for (sidx, seed) in enumerate(seeds)
        if sidx % 10 == 0 println("seed: $(sidx)") end
        reset!(col, seed)
        total_density += compute_density(col.scene, col.roadway)
    end
    avg_density = total_density / length(seeds)
    return avg_density
end

dist(a::VecSE2, b::VecSE2) = hypot(a.x-b.x, a.y-b.y)

function compute_total_distance_traveled(start::Scene, final::Scene)
    total_dist = 0
    for (start_veh, final_veh) in zip(start, final)
        total_dist += dist(final_veh.state.posG, start_veh.state.posG)
    end
    return total_dist
end

function compute_heuristic_mean_travel_distance(
        flags::Flags, seeds::Vector{Int}, T::Float64 = 20.)
    col = build_dataset_collector("", flags)
    prev_scene = Scene()
    total_dist = 0
    for (sidx, seed) in enumerate(seeds)
        reset!(col, seed)
        actions = Array(DriveAction, length(col.scene))
        seed_dist = 0
        for t in 0:col.eval.rec.timestep:(T - col.eval.rec.timestep)
            copy!(prev_scene, col.scene)
            get_actions!(actions, col.scene, col.roadway, col.models)
            tick!(col.scene, col.roadway, actions, col.models)
            seed_dist += compute_total_distance_traveled(
                prev_scene, col.scene)
        end
        total_dist += seed_dist
        println("seed: $(sidx)\tdist: $(seed_dist)\ttotal dist: $(total_dist)\tavg dist: $(total_dist / sidx)")
    end
    avg_dist = total_dist / length(seeds)
    return avg_dist
end

# parse flags
parse_flags!(FLAGS, ARGS)

# compute heuristic density
# seeds = collect(1:1000)
# avg_density = compute_heuristic_density(FLAGS, seeds)
# println(avg_density)

# compute travel distance
FLAGS["sampling_period"] = .5
seeds = collect(1:200)
avg_dist = compute_heuristic_mean_travel_distance(FLAGS, seeds)
println(avg_dist)


