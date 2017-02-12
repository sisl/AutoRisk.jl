# using Base.Test
# using AutoRisk
# using AutomotiveDrivingModels

# push!(LOAD_PATH, "../../../src")
# include("../../../src/generation/behavior/parameters.jl")
# include("../../../src/generation/behavior/behavior_generator.jl")
# include("../../../src/generation/behavior/heuristic_behavior_generators.jl")

# include("../../../src/utils/automotive.jl")

function test_predefined_behavior_generator()
    idm_params = IDMParams(collect(0.:7.)...,0.)
    mobil_params = MOBILParams(collect(8.:10.)...)
    lat_params = LateralParams(11., 12., 12.)
    lat_params_2 = LateralParams(13., 14., 14.)
    params = [BehaviorParams(idm_params, mobil_params, lat_params),
        BehaviorParams(idm_params, mobil_params, lat_params_2)]
    weights = WeightVec([.5, .5])
    context = IntegratedContinuous(.1, 1)
    gen = PredefinedBehaviorGenerator(context, params, weights)

    srand(1)
    samp_params = rand(gen)
    @test samp_params == params[1]
    srand(3)
    samp_params = rand(gen)
    @test samp_params == params[2]
end

function test_predefined_behavior_generator_non_determinism()
    
    # nondeterministic case
    params = get_normal_behavior_params(lon_σ = 1., lat_σ = .1)
    context = IntegratedContinuous(.1, 1)
    num_vehicles = 1 
    driver = build_driver(params, context, num_vehicles)

    roadway = gen_straight_roadway(1)
    scene = Scene(1)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    veh_state = VehicleState(Frenet(road_idx, roadway), 
        roadway, 10.)
    veh_def = VehicleDef(1, AgentClass.CAR, 1., 1.)
    veh = Vehicle(veh_state, veh_def)
    push!(scene, veh)
    observe!(driver, scene, roadway, 1)
    
    srand(1)
    act_1 = rand(driver)

    observe!(driver, scene, roadway, 1)
    srand(1)
    act_2 = rand(driver)

    @test act_1 == act_2

    srand(2)
    observe!(driver, scene, roadway, 1)
    observe!(driver, scene, roadway, 1)
    observe!(driver, scene, roadway, 1)
    observe!(driver, scene, roadway, 1)
    observe!(driver, scene, roadway, 1)
    act_3 = rand(driver)
    @test act_1 != act_3

    # deterministic case
    params = get_normal_behavior_params()
    driver = build_driver(params, context, num_vehicles)
    observe!(driver, scene, roadway, 1)

    srand(1)
    act_1 = rand(driver)

    observe!(driver, scene, roadway, 1)
    srand(2)
    act_2 = rand(driver)

    @test act_1 == act_2
end

function test_uniform_behavior_generator()
    idm_params = IDMParams(collect(0.:7.)..., 0.)
    mobil_params = MOBILParams(collect(8.:10.)...)
    lat_params = LateralParams(11., 12., 13.)
    min_params = BehaviorParams(idm_params, mobil_params, lat_params)
    idm_params = IDMParams(collect(1.:8.)...,0.)
    mobil_params = MOBILParams(collect(9.:11.)...)
    lat_params = LateralParams(12., 13., 14.)
    max_params = BehaviorParams(idm_params, mobil_params, lat_params)
    context = IntegratedContinuous(.1, 1)
    gen = UniformBehaviorGenerator(context, min_params, min_params)
    srand(1)
    samp_params_min = rand(gen)
    @test samp_params_min == min_params

    gen = UniformBehaviorGenerator(context, max_params, max_params)
    samp_params_max = rand(gen)
    @test samp_params_max == max_params

    gen = UniformBehaviorGenerator(context, min_params, max_params)
    samp_params = rand(gen)
    @test samp_params != samp_params_min
    @test samp_params != samp_params_max
end

@time test_predefined_behavior_generator()
@time test_predefined_behavior_generator_non_determinism()
@time test_uniform_behavior_generator()
