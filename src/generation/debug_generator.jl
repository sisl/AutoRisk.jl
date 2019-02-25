export 
    DebugSceneGenerator,
    DebugBehaviorGenerator,
    rand!

@with_kw mutable struct DebugSceneGenerator <: SceneGenerator
    # Δs = (fore_s - rear_s)
    lo_Δs::Float64 = 0.
    hi_Δs::Float64 = 10.
    # abs quantities
    lo_v_rear::Float64 = 1.
    hi_v_rear::Float64 = 1.
    lo_v_fore::Float64 = 0.
    hi_v_fore::Float64 = 0.
    # step for the random values (set to the range to choose between end points)
    s_eps::Float64 = 1e-8
    v_eps::Float64 = 1e-8

    rng::MersenneTwister = MersenneTwister(1)
end
function Random.rand!(gen::DebugSceneGenerator, scene::Scene, 
        roadway::Roadway, seed::Int64) 
    # set random seed
    srand(gen.rng, seed)

    # remove old contents of scene
    empty!(scene)

    # sample random values
    Δs = rand(gen.rng, gen.lo_Δs:gen.s_eps:gen.hi_Δs)
    rear_v = rand(gen.rng, gen.lo_v_rear:gen.v_eps:gen.hi_v_rear)
    fore_v = rand(gen.rng, gen.lo_v_fore:gen.v_eps:gen.hi_v_fore)    

    # rear vehicle (veh id 1)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    rear_state = VehicleState(Frenet(road_idx, roadway), roadway, rear_v)
    rear_def = VehicleDef(AgentClass.CAR, 2., 1.)
    push!(scene, Vehicle(rear_state, rear_def, 1))

    # fore vehicle (veh id 2)
    fore_state = VehicleState(Frenet(road_idx, roadway), roadway, fore_v)
    fore_state = move_along(fore_state, roadway, Δs)
    fore_def = VehicleDef(AgentClass.CAR, 2., 1.)
    push!(scene, Vehicle(fore_state, fore_def, 2))

    return scene
end

@with_kw mutable struct DebugBehaviorGenerator <: BehaviorGenerator
    Δt::Float64 = .1
    rear_lon_σ::Float64 = 0.0
    fore_lon_σ::Float64 = 0.0  
    rng::MersenneTwister = MersenneTwister(1)
end
function Random.rand!(gen::DebugBehaviorGenerator, models::Dict{Int, DriverModel}, 
        scene::Scene, seed::Int64)
    # zero acceleration models with variable std dev
    models[1] = Tim2DDriver(gen.Δt, mlon = StaticLongitudinalDriver(
        0.0, gen.rear_lon_σ))
    models[2] = Tim2DDriver(gen.Δt, mlon = StaticLongitudinalDriver(
        0.0, gen.fore_lon_σ))
    return models
end
