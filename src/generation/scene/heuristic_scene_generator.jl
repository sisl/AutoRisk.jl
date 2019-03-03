export 
    HeuristicSceneGenerator,
    generate_init_road_idxs,
    is_valid_position,
    permits_valid_positions,
    generate_road_positions,
    build_vehicle,
    rand!

"""
# Description:
    - HeuristicSceneGenerator generates scenes based on heursitics - i.e., it 
        just uses rules that align with expectation, but that do not reflect 
        any learned model, for example. 
"""
mutable struct HeuristicSceneGenerator <: SceneGenerator
    min_num_vehicles::Int64
    max_num_vehicles::Int64

    min_base_speed::Float64
    max_base_speed::Float64

    min_vehicle_length::Float64
    max_vehicle_length::Float64

    min_vehicle_width::Float64
    max_vehicle_width::Float64

    min_init_dist::Float64
    
    # optional
    max_init_dist::Float64
    rng::MersenneTwister
    total_roadway_length::Float64
    mode::String

    function HeuristicSceneGenerator(min_num_vehicles, max_num_vehicles, 
            min_base_speed, max_base_speed, min_vehicle_length, 
            max_vehicle_length, min_vehicle_width,  max_vehicle_width, 
            min_init_dist;  max_init_dist = 0, rng = MersenneTwister(1),
            total_roadway_length = 0, mode = "const_spaced")
        return new(min_num_vehicles, max_num_vehicles, 
            min_base_speed, max_base_speed, min_vehicle_length, 
            max_vehicle_length, min_vehicle_width,  max_vehicle_width, 
            min_init_dist,  max_init_dist, rng, 
            total_roadway_length, mode)
    end
end

"""
# Description:
    - Generate RoadIndex objects for num_vehicles

# Args:
    - gen: generator to use
    - roadway: roadway from which to derive indices
    - num_vehicles: number of road idxs to generate

# Returns:
    - vector of road indices
"""
function generate_init_road_idxs(gen::HeuristicSceneGenerator, 
        roadway::Roadway, num_vehicles::Int64)
    init_road_idxs = Vector{RoadIndex}(undef, num_vehicles)

    # use the number of lanes in the first roadway segment
    num_lanes = length(roadway.segments[1].lanes)

    # use the width of the first lane in the first segment
    lane_width = roadway.segments[1].lanes[1].width

    # precompute possible offsets
    offsets = [l * lane_width for l in 0:(num_lanes - 1)]

    # if stadium roadway, then make offsets negative
    if length(roadway.segments) != 1
        offsets *= -1
    end

    # loop through vehicles, randomly assigning road indices
    for idx in 1:num_vehicles
        lane_offset = rand(gen.rng, offsets)
        road_idx = RoadIndex(proj(VecSE2(0.0, lane_offset, 0.0), roadway))
        init_road_idxs[idx] = road_idx
    end
    return init_road_idxs
end

"""
# Description:
    - Determines whether a position is valid given the positions of cars so far
        placed on the roadway

# Args:
    - gen: generator to use
    - pos: position being assessed
    - positions: already-existing positions
    - lanes: lanes of the corresponding positions

# Returns:
    - whether or not the position is valid
"""
function is_valid_position(gen::HeuristicSceneGenerator, pos::Float64, 
        positions::Vector{Float64}, lanes::Vector{Int64})
    valid = true
    cur_idx = length(positions) + 1
    for (idx, other_pos) in enumerate(positions)
        # skip if lanes differ
        if lanes[cur_idx] == lanes[idx]

            # need to account for wrap around, though
            # assume all the individual values are positive
            dist = abs(pos - other_pos)

            # e.g., if roadway_length = 10
            # pos1 = 1, pos2 = 9, dist = 8
            # then true_dist = 10 - 8 = 2
            dist = min(dist, gen.total_roadway_length - dist)
            if dist < gen.min_init_dist
                valid = false
                break
            end
        end
    end
    return valid
end

function permits_valid_positions(gen::HeuristicSceneGenerator, 
        lanes::Vector{Int64})
    # check whether a valid set of positions is possible
    counts = Dict()
    for l in lanes
        if in(l, keys(counts))
            counts[l] += gen.min_init_dist
        else
            counts[l] = gen.min_init_dist
        end
    end
    for c in values(counts)
        if c > gen.total_roadway_length
            return false
        end
    end
    return true
end

"""
# Description:
    - Generates road positions for vehicles based on their lanes

# Args:
    - gen: generator to use
    - lanes: lanes of the vehicles to generate

# Returns:
    - positions
"""
function generate_road_positions(gen::HeuristicSceneGenerator, 
        lanes::Vector{Int64}, roadway_type::String = "stadium")
    # check if valid placement possible
    if !permits_valid_positions(gen, lanes)
        throw(ArgumentError("Too many positions requested: \n$(lanes)"))
    end

    # select random position on roadway for ego vehicle
    if roadway_type == "stadium"
        ego_x = gen.total_roadway_length * rand(gen.rng)
    else # straight roadway
        ego_x = gen.total_roadway_length * .1
    end

    # randomly generate locations for other vehicles around ego vehicle
    positions = Vector{Float64}()
    push!(positions, ego_x)
    for idx in 2:length(lanes)
        valid = false
        while !valid
            # place this car in front or behind
            sign = rand(gen.rng) > .5 ? 1 : -1

            # select distance
            dist = gen.min_init_dist + rand(gen.rng) * (
                gen.max_init_dist - gen.min_init_dist)

            # compute position and check if valid
            pos = (sign * dist + ego_x) % gen.total_roadway_length
            # if negative position then wrap around
            if pos < 0
                pos = gen.total_roadway_length + pos
            end
            valid = is_valid_position(gen, pos, positions, lanes)

            # if invalid increase variance in placement to avoid
            # infinite loop in many car scenario
            if !valid
                gen.max_init_dist += 1
            else
                push!(positions, pos)
            end
        end
    end
    return positions
end


"""
# Description:
    - Generates road positions for vehicles based on their lanes

# Args:
    - gen: generator to use
    - lanes: lanes of the vehicles to generate

# Returns:
    - positions
"""
function generate_const_spaced(gen::HeuristicSceneGenerator, 
            num_vehicles::Int, roadway::Roadway)
    num_lanes = nlanes(roadway) 
    lane_width = roadway.segments[1].lanes[1].width
    # precompute possible offsets
    offsets = [-l * lane_width for l in 0:(num_lanes - 1)]
    num_veh_per_lane = Int(ceil(num_vehicles / num_lanes))
    # distance between vehicles to maximally space around track
    spacing = (gen.total_roadway_length - gen.min_init_dist) / num_veh_per_lane
    # check if valid placement possible
    if gen.min_init_dist > spacing
        throw(ArgumentError(
            "Roadway too small for requested num vehicles: $(num_vehicles)"))
    end

    road_idxs = RoadIndex[]
    positions = Float64[]

    for lane_idx in 1:num_lanes
        lane_offset = offsets[lane_idx]
        pos_x = 0
        if lane_idx == num_lanes
            # last lane only generates remaining vehicles
            num_veh_per_lane = num_vehicles - length(road_idxs)
            spacing = (gen.total_roadway_length - gen.min_init_dist) / num_veh_per_lane
        end
        for _ in 1:num_veh_per_lane
            push!(positions, pos_x)
            push!(road_idxs, RoadIndex(proj(VecSE2(0.0, lane_offset, 0.0), roadway)))
            pos_x += spacing
        end
    end
    return road_idxs, positions
end

"""
# Description:
    - Builds a vehicle.

# Args:
    - gen: generator to use
    - roadway: on which to place vehicle
    - road_idx: of the vehicle to generate
    - veh_id: of the vehicle to generate

# Returns:
    - a vehicle
"""
function build_vehicle(gen::HeuristicSceneGenerator, roadway::Roadway, 
        road_idx::RoadIndex, road_pos::Float64, veh_id::Int64)

    # build and move vehicle state
    base_speed = gen.min_base_speed + rand(gen.rng) * (
        gen.max_base_speed - gen.min_base_speed)
    veh_state = VehicleState(Frenet(road_idx, roadway), 
        roadway, base_speed)
    veh_state = move_along(veh_state, roadway, road_pos)

    # build vehicle definition and vehicle
    base = rand(gen.rng)
    width_base = (base + rand(gen.rng)) / 2.
    length_base = (base + rand(gen.rng)) / 2.
    veh_length = gen.min_vehicle_length + length_base * (
        gen.max_vehicle_length - gen.min_vehicle_length)
    veh_width = gen.min_vehicle_width + width_base * (
        gen.max_vehicle_width - gen.min_vehicle_width)
    veh_def = VehicleDef(AgentClass.CAR, veh_length, veh_width)

    return Vehicle(veh_state, veh_def, veh_id)
end

"""
# Description:
    - Reset a scene using the generator.

# Args:
    - gen: generator to use
    - scene: scene to populate with vehicles.
    - roadway: on which to place vehicles
    - seed: random seed to use for generation
"""
function Random.rand!(gen::HeuristicSceneGenerator, scene::Scene, 
        roadway::Roadway, seed::Int64) 
    # set random seed
    srand(gen.rng, seed)

    # heuristic generator assumes stadium roadway
    gen.total_roadway_length = get_total_roadway_length(roadway)
    if get_roadway_type(roadway) == "straight"
        gen.max_init_dist = gen.min_init_dist # gen.total_roadway_length / 10.
    else
        gen.max_init_dist = gen.total_roadway_length / 2 - gen.min_init_dist
    end

    # remove old contents of scene and models
    empty!(scene)

    num_vehicles = rand(gen.rng, gen.min_num_vehicles:gen.max_num_vehicles)

    # get initial road indices, positions, vehicles
    if gen.mode == "const_spaced"
        init_road_idxs, road_positions = generate_const_spaced(
            gen, num_vehicles, roadway)
    else
        init_road_idxs = generate_init_road_idxs(gen, roadway, num_vehicles)
        lanes = [road_idx.tag.lane for road_idx in init_road_idxs]
        road_positions = generate_road_positions(
            gen, lanes, get_roadway_type(roadway))
    end
    
    # add vehicles to scene
    for (idx, (road_idx, road_pos)) in enumerate(
            zip(init_road_idxs, road_positions))
        push!(scene, build_vehicle(gen, roadway, road_idx, road_pos, idx)) 
    end
    return scene
end
