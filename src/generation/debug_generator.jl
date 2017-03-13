export 
    DebugSceneGenerator,
    DebugBehaviorGenerator,
    rand!

@with_kw type DebugSceneGenerator <: SceneGenerator
    # Δs = (fore_s - rear_s)
    lo_Δs::Float64 = 0.
    hi_Δs::Float64 = 10.
    # Δv = (fore_v - rear_v)
    lo_Δv::Float64 = -1.
    hi_Δv::Float64 = 1.
    # abs quantities of rear vehicle
    lo_v::Float64 = 0.
    hi_v::Float64 = 5.

    rng::MersenneTwister = MersenneTwister(1)
end
function Base.rand!(gen::DebugSceneGenerator, scene::Scene, 
        roadway::Roadway, seed::Int64) 
    # set random seed
    srand(gen.rng, seed)

    # remove old contents of scene
    empty!(scene)

    # sample random values
    eps = 1e-8
    Δs = rand(gen.rng, gen.lo_Δs:eps:gen.hi_Δs)
    Δv = rand(gen.rng, gen.lo_Δv:eps:gen.hi_Δv)
    rear_v = rand(gen.rng, gen.lo_v:eps:gen.hi_v)

    # rear vehicle (veh id 1)
    road_idx = RoadIndex(proj(VecSE2(0.0, 0.0, 0.0), roadway))
    rear_state = VehicleState(Frenet(road_idx, roadway), roadway, rear_v)
    rear_def = VehicleDef(1, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(rear_state, rear_def))

    # fore vehicle (veh id 2)
    fore_v = rear_v + Δv
    fore_state = VehicleState(Frenet(road_idx, roadway), roadway, fore_v)
    fore_state = move_along(fore_state, roadway, Δs)
    fore_def = VehicleDef(2, AgentClass.CAR, 5., 2.)
    push!(scene, Vehicle(fore_state, fore_def))

    return scene
end

@with_kw type DebugBehaviorGenerator <: BehaviorGenerator
    context::ActionContext = IntegratedContinuous(.1, 1)
    rear_lon_σ::Float64 = 0.0
    fore_lon_σ::Float64 = 0.0  
    rng::MersenneTwister = MersenneTwister(1)
end
function Base.rand!(gen::DebugBehaviorGenerator, models::Dict{Int, DriverModel}, 
        scene::Scene, seed::Int64)
    # zero acceleration models with variable std dev
    models[1] = Tim2DDriver(gen.context, mlon = StaticLongitudinalDriver(
        0.0, gen.rear_lon_σ))
    models[2] = Tim2DDriver(gen.context, mlon = StaticLongitudinalDriver(
        0.0, gen.fore_lon_σ))
    models
end