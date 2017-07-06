# using Base.Test
# using AutoRisk

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
        rng = MersenneTwister(1),
        mode = "ego")
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
        max_init_dist = max_init_dist,
        rng = rng,
        mode = mode)
end

function test_heuristic_scene_generator_constructor()
    gen = build_debug_heuristic_scene_generator()
end

function test_generate_init_road_idxs()
    gen = build_debug_heuristic_scene_generator()
    gen.total_roadway_length = 200. + 50. * pi * 2 
    roadway = gen_stadium_roadway(1)
    num_vehicles = 5
    init = generate_init_road_idxs(gen, roadway, num_vehicles)
    @test length(init) == num_vehicles

    gen = build_debug_heuristic_scene_generator()
    roadway = gen_stadium_roadway(10, length = 1000., radius = 50.)
    gen.total_roadway_length = get_total_roadway_length(roadway)
    num_vehicles = 100
    init = generate_init_road_idxs(gen, roadway, num_vehicles)
    lanes = [road_idx.tag.lane for road_idx in init]
    @test length(init) == num_vehicles
    @test Set(lanes) == Set(1:10)

    gen = build_debug_heuristic_scene_generator()
    roadway = gen_straight_roadway(2, 100., lane_width = 3.)
    gen.total_roadway_length = get_total_roadway_length(roadway)
    num_vehicles = 10
    init = generate_init_road_idxs(gen, roadway, num_vehicles)
    lanes = [road_idx.tag.lane for road_idx in init]
    @test length(init) == num_vehicles
    @test Set(lanes) == Set(1:2)
end

function test_is_valid()
    gen = build_debug_heuristic_scene_generator()
    roadway = gen_stadium_roadway(1, length = 200., radius = 25.)
    gen.total_roadway_length = 200. + 25. * pi * 2 
    
    # basic, two car tests
    lanes = [1,1]
    pos = 10.
    positions = [pos]
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == false

    lanes = [1,2]
    pos = 10.
    positions = [pos]
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == true

    lanes = [1,1]
    positions = [pos + gen.min_init_dist]
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == true

    positions = [pos - gen.min_init_dist]
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == true

    positions = [pos - gen.min_init_dist + 1]
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == false

    lanes = [1,1]
    pos = gen.total_roadway_length
    positions = [gen.min_init_dist - 1]
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == false

    lanes = [1,1]
    pos = gen.total_roadway_length
    positions = [gen.min_init_dist + 1]
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == true

    # multicar tests
    lanes = [1,2,1,1,1,1,1]
    pos = 50.
    positions = [5., 0., 15., 25., 35., 45.]
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == false

    pos = 55.
    valid = is_valid_position(gen, pos, positions, lanes)
    @test valid == true
end

function test_generate_road_positions()
    roadway = gen_stadium_roadway(2, length = 100., radius = 50.)
    lanes = [1,2,2]
    gen = build_debug_heuristic_scene_generator(min_init_dist = 20., 
        max_init_dist = 30.)
    gen.total_roadway_length = 200. + 50. * pi * 2 

    actual_positions = generate_road_positions(gen, lanes)
    for (p1, p2) in zip(actual_positions, actual_positions[2:end])
        @test abs(p1 - p2) > gen.min_init_dist
    end

    lanes = ones(Int64, 10)
    actual_positions = generate_road_positions(gen, lanes)
    sort!(actual_positions)
    for (p1, p2) in zip(actual_positions, actual_positions[2:end])
        @test abs(p1 - p2) > gen.min_init_dist
    end
end

function test_build_vehicle()
    min_num_vehicles, max_num_vehicles = 4, 100
    min_base_speed, max_base_speed = 10., 40.
    min_vehicle_length, max_vehicle_length = 3., 10.
    min_vehicle_width, max_vehicle_width = 1., 3.
    
    gen = build_debug_heuristic_scene_generator(
        min_num_vehicles=min_num_vehicles, 
        max_num_vehicles=max_num_vehicles,
        min_base_speed=min_base_speed,
        max_base_speed=max_base_speed,
        min_vehicle_length=min_vehicle_length,
        max_vehicle_length=max_vehicle_length,
        min_vehicle_width=min_vehicle_width, 
        max_vehicle_width=max_vehicle_width)
    roadway_length = 100.
    roadway = gen_stadium_roadway(2, length = roadway_length, width = 50.)
    road_idx = RoadIndex(proj(VecSE2(0.0, -4.0, 0.0), roadway))
    road_pos = 10.
    veh_id = 3
    veh = build_vehicle(gen, roadway, road_idx, road_pos, veh_id)
    eps = 1e-2
    @test min_base_speed <= veh.state.v <= max_base_speed
    @test veh.state.posF.roadind.tag.segment == 1
    @test veh.state.posF.roadind.tag.lane == 1
    @test abs(veh.state.posF.s - (road_pos - roadway_length)) < eps
    @test abs(veh.state.posF.t) < eps
    @test veh.state.posF.ϕ == 0.
    @test veh.id == veh_id
    @test min_vehicle_length <= veh.def.length <= max_vehicle_length
    @test min_vehicle_width <= veh.def.width <= max_vehicle_width
    @test veh.def.class == AgentClass.CAR
end

function test_heuristic_scene_reset()
    roadway = gen_stadium_roadway(5, length = 100., radius = 50.)
    gen = build_debug_heuristic_scene_generator(min_init_dist = 10., 
        max_init_dist = 20., max_num_vehicles = 5)

    # check same seed gives same scene
    scene_1 = Scene(5)
    seed = 1
    rand!(gen, scene_1, roadway, seed)
    scene_2 = Scene(5)
    seed = 1
    rand!(gen, scene_2, roadway, seed)
    @test scene_1 == scene_2
   
end

function test_const_spaced_mode()
    roadway = gen_stadium_roadway(5, length = 100., radius = 50.)
    num_vehicles = 20
    gen = build_debug_heuristic_scene_generator(min_init_dist = 10., 
        min_num_vehicles = num_vehicles, max_num_vehicles = num_vehicles, 
        mode = "const_spaced")
    seed = 1
    scene = rand!(gen, Scene(num_vehicles), roadway, seed)
    println(scene.entities[1])
    println(scene.entities[end])
end

@time test_heuristic_scene_generator_constructor()
@time test_generate_init_road_idxs()
@time test_is_valid()
@time test_generate_road_positions()
@time test_build_vehicle()
@time test_heuristic_scene_reset()
@time test_const_spaced_mode()
