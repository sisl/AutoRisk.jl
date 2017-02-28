# using Base.Test
# using AutomotiveDrivingModels

# push!(LOAD_PATH, "../../../src")
# include("../../../src/generation/behavior/parameters.jl")
# include("../../../src/generation/behavior/behavior_generator.jl")
# include("../../../src/generation/behavior/heuristic_behavior_generators.jl")
# include("../../../src/utils/automotive.jl")

function test_build_driver()
    σ = 0.
    k_spd = 1.
    δ = 2.
    T = 3. 
    v_des = 4. 
    s_min = 5.
    a_max = 6.
    d_cmf = 7.
    idm_params = IDMParams(σ, k_spd, δ, T, v_des, s_min, a_max, d_cmf, 0.0)
    politeness = 8.
    safe_decel = 9.
    advantage_threshold = 10.
    mobil_params = MOBILParams(politeness, safe_decel, advantage_threshold)
    kp = 11.
    kd = 12.
    lat_params = LateralParams(σ, kp, kd)
    params = BehaviorParams(idm_params, mobil_params, lat_params)
    context = IntegratedContinuous(.1, 1)
    num_vehicles = 13
    driver = build_driver(params, context, num_vehicles)
    @test driver.mlon.k_spd == k_spd
    @test driver.mlon.δ == δ
    @test driver.mlon.v_des == v_des
    @test driver.mlon.s_min == s_min
    @test driver.mlon.a_max == a_max
    @test driver.mlon.d_cmf == d_cmf
    @test driver.mlane.politeness == politeness
    @test driver.mlane.advantage_threshold == advantage_threshold
    @test driver.mlane.safe_decel == safe_decel
    @test driver.mlat.kp == kp 
    @test driver.mlat.kd == kd
end

function test_behavior_reset()

    idm_params = IDMParams(0.,1.,2.,3.,4.,5.,6.,7.,0.)
    mobil_params = MOBILParams(collect(8.:10.)...)
    lat_params = LateralParams(11., 12., 13.)
    params = [BehaviorParams(idm_params, mobil_params, lat_params)]
    weights = WeightVec([1.])
    context = IntegratedContinuous(.1, 1)
    gen = PredefinedBehaviorGenerator(context, params, weights)
    models = Dict{Int, DriverModel}()
    scene = Scene(1)
    seed = 1
    rand!(gen, models, scene, seed)
end


@time test_build_driver()
@time test_behavior_reset()
