# using Base.Test
# using AutoRisk

function test_predefined_behavior_generator()
    idm_params = IDMParams(collect(0.:7.)...)
    mobil_params = MOBILParams(collect(8.:10.)...)
    lat_params = LateralParams(11., 12., 12.)
    lat_params_2 = LateralParams(13., 14., 14.)
    params = [BehaviorParams(idm_params, mobil_params, lat_params),
        BehaviorParams(idm_params, mobil_params, lat_params_2)]
    weights = StatsBase.Weights([.5, .5])
    gen = PredefinedBehaviorGenerator(params, weights)

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
    num_vehicles = 1 
    driver = build_driver(params, num_vehicles)

    roadway = gen_straight_roadway(1)
    scene = Scene(1)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    veh_state = VehicleState(Frenet(road_idx, roadway), 
        roadway, 10.)
    veh_def = VehicleDef(AgentClass.CAR, 1., 1.)
    veh = Vehicle(veh_state, veh_def, 1)
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
    driver = build_driver(params, num_vehicles)
    observe!(driver, scene, roadway, 1)

    srand(1)
    act_1 = rand(driver)

    observe!(driver, scene, roadway, 1)
    srand(2)
    act_2 = rand(driver)

    @test act_1 == act_2
end

function test_uniform_behavior_generator()
    idm_params = IDMParams(collect(0.:7.)...)
    mobil_params = MOBILParams(collect(8.:10.)...)
    lat_params = LateralParams(11., 12., 13.)
    min_params = BehaviorParams(idm_params, mobil_params, lat_params)
    idm_params = IDMParams(collect(1.:8.)...)
    mobil_params = MOBILParams(collect(9.:11.)...)
    lat_params = LateralParams(12., 13., 14.)
    max_params = BehaviorParams(idm_params, mobil_params, lat_params)
    gen = UniformBehaviorGenerator(min_params, min_params)
    srand(1)
    samp_params_min = rand(gen)
    @test samp_params_min == min_params

    gen = UniformBehaviorGenerator(max_params, max_params)
    samp_params_max = rand(gen)
    @test samp_params_max == max_params

    gen = UniformBehaviorGenerator(min_params, max_params)
    samp_params = rand(gen)
    @test samp_params != samp_params_min
    @test samp_params != samp_params_max
end

function test_correlated_behavior_generator()
    min_p = get_passive_behavior_params()
    max_p = get_aggressive_behavior_params()
    gen = CorrelatedBehaviorGenerator(min_p, max_p)
    srand(gen.rng, 1)
    params_1 = rand(gen)
    srand(gen.rng, 1)
    params_2 = rand(gen)
    srand(gen.rng, 2)
    params_3 = rand(gen)
    @test params_1 == params_2
    @test params_2 != params_3
end

function test_correlated_gaussian_behavior_generator()
    min_p = get_passive_behavior_params()
    max_p = get_aggressive_behavior_params()
    gen = CorrelatedGaussianBehaviorGenerator(min_p, max_p)
    srand(gen.rng, 1)
    params_1 = rand(gen)
    srand(gen.rng, 1)
    params_2 = rand(gen)
    srand(gen.rng, 2)
    params_3 = rand(gen)
    @test params_1 == params_2
    @test params_2 != params_3
end

function test_truncated_gaussian_sample()
    mu = 0.
    σ = 1.
    high = .5
    low = -5.
    rng = MersenneTwister()
    n_samples = 1000
    for i in 1:n_samples
        x = tuncated_gaussian_sample(rng, mu, σ, high, low)
        @test x > low && x < high
    end
end

function test_truncated_gaussian_sample_from_agg()
    rng = MersenneTwister()
    relative_σ = .1
    n_samples = 10000

    high = 1.3
    low = -.7
    xs = zeros(n_samples)
    for i in 1:n_samples
        agg = rand()
        xs[i] = truncated_gaussian_sample_from_agg(rng, agg, relative_σ, high, low)
        @test xs[i] > low && xs[i] < high
    end

    high = -1.3
    low = .7
    xs = zeros(n_samples)
    for i in 1:n_samples
        agg = rand()
        xs[i] = truncated_gaussian_sample_from_agg(rng, agg, relative_σ, high, low, flip=true)
        @test xs[i] < low && xs[i] > high
    end
end

@time test_predefined_behavior_generator()
@time test_predefined_behavior_generator_non_determinism()
@time test_uniform_behavior_generator()
@time test_correlated_behavior_generator()
@time test_truncated_gaussian_sample()
@time test_truncated_gaussian_sample_from_agg()
@time test_correlated_gaussian_behavior_generator()