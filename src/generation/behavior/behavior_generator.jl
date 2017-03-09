export 
    BehaviorGenerator,
    rand!,
    build_driver

"""
# Description:
    - BehaviorGenerator is the abstract type underlying the behavior generators.
"""
abstract BehaviorGenerator

Base.rand(gen::BehaviorGenerator) = error("rand not implemented for $(gen)")

"""
# Description:
    - This method uses any behavior generator to populate a models dict with 
        driver models

# Args:
    - gen: behavior generator to use
    - models: dict to populate
    - scene: scene that contains vehicles which correspond to the driver models
    - seed: random seed to use in populating the models
"""
function Base.rand!(gen::BehaviorGenerator, models::Dict{Int, DriverModel}, 
        scene::Scene, seed::Int64)
    srand(seed)
    srand(gen.rng, seed)
    empty!(models)
    for veh in scene.vehicles
        params = rand(gen)
        models[veh.def.id] = build_driver(params, gen.context, length(scene))
        srand(models[veh.def.id], seed)
    end
end

"""
# Description:
    - Builds a driver using provided parameter set. This assumes usage of 
        IDM, MOBIL, and ProportionalLaneTracker.

# Args:
    - p: params to use
    - context: context for driver to use
    - num_vehicles: number of vehicles in the scene
"""
function build_driver(p::BehaviorParams, context::IntegratedContinuous,
        num_vehicles::Int64)
    if p.lon_response_time != 0.0
        mlon = DelayedIntelligentDriverModel(context.Δt,
            k_spd = p.idm.k_spd,
            δ = p.idm.δ,
            T = p.idm.T,
            v_des = p.idm.v_des,
            s_min = p.idm.s_min,
            a_max = p.idm.a_max,
            d_cmf = p.idm.d_cmf,
            t_d = p.lon_response_time)
    else
        mlon = IntelligentDriverModel(
            k_spd = p.idm.k_spd,
            δ = p.idm.δ,
            T = p.idm.T,
            v_des = p.idm.v_des,
            s_min = p.idm.s_min,
            a_max = p.idm.a_max,
            d_cmf = p.idm.d_cmf)
    end
    mlon.σ = p.idm.σ
    mlat = ProportionalLaneTracker(
        σ = p.lat.σ,
        kp = p.lat.kp, 
        kd = p.lat.kd)
    mlat.σ = p.lat.σ
    mlane = MOBIL(context,
        safe_decel = p.mobil.safe_decel,
        politeness = p.mobil.politeness,
        advantage_threshold = p.mobil.advantage_threshold)
    model = Tim2DDriver(context, 
        rec = SceneRecord(1, context.Δt, num_vehicles), 
        mlat = mlat, 
        mlon = mlon, 
        mlane = mlane)
    if p.overall_response_time != 0.0
        model = DelayedDriver(model, reaction_time = p.overall_response_time)
    end
    if p.err_p_a_to_i != 0.0
        model = ErrorableDriverModel(model, p_a_to_i = p.err_p_a_to_i, 
            p_i_to_a = p.err_p_i_to_a)
    end
    set_desired_speed!(model, p.idm.v_des)
    return model
end
