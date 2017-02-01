
using AutoRisk
using NGSIM

function compute_ngsim_density(trajdata, step = 100)
    scene = Scene()
    num_frames = length(trajdata.frames)
    num_scenes = length(1:step:num_frames)
    total_density = 0
    roadway_area = 640. * 3. * 5.
    for fidx in 1:step:num_frames
        get!(scene, trajdata, fidx)
        total_density += scene.n_vehicles / roadway_area
    end
    return total_density / num_scenes
end

function compute_num_unique_vehicles(trajdata)
    ids = Set()
    scene = Scene()
    for fidx in 1:length(trajdata.frames)
        get!(scene, trajdata, fidx)
        for vehicle in scene
            push!(ids, vehicle.def.id)
        end
    end
    return length(ids)
end

function compute_ngsim_targets(trajdata)
    scene = Scene()
    rec = SceneRecord(10000, .1)
    roadway = ROADWAY_101
    targets = Array{Float64}(2, 10000)
    fill!(targets, 0)
    veh_id_to_idx = Dict{Int64, Int64}()
    done = Set{Int}()
    num_frames = length(trajdata.frames)
    totals = zeros(2,1)
    for fidx in 1:num_frames
        if fidx % 100 == 0
            println("$(fidx) / $(num_frames)\t$(totals)")
        end
        get!(scene, trajdata, fidx)
        update!(rec, scene)
        get_veh_id_to_idx(scene, veh_id_to_idx)
        extract_frame_targets!(rec, roadway, targets, veh_id_to_idx,
            true, done, 0)
        totals += sum(targets, 2)
        empty!(veh_id_to_idx)
        fill!(targets, 0)
    end
    return totals
end

# prelim
trajdata = load_trajdata(1)

# density
# avg_density = compute_ngsim_density(trajdata)
# println(avg_density)

# number of unique vehicles
num_veh_total = compute_num_unique_vehicles(trajdata)
println(num_veh_total)

# targets 
# target_sums = compute_ngsim_targets(trajdata)
# println(target_sums)