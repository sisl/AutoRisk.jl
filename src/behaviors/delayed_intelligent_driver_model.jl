export 
    DelayedIntelligentDriverModel,
    get_name,
    set_desired_speed!,
    track_longitudinal!,
    observe!,
    rand,
    pdf,
    logpdf

import AutomotiveDrivingModels: 
    LaneFollowingDriver, 
    get_name,
    set_desired_speed!,
    track_longitudinal!,
    observe!

"""
IDM with a reaction time delay
"""
type DelayedIntelligentDriverModel <: LaneFollowingDriver
    a::Float64 # predicted acceleration
    σ::Float64 # optional stdev on top of the model, set to zero or NaN for deterministic behavior

    k_spd::Float64 # proportional constant for speed tracking when in freeflow [s⁻¹]

    δ::Float64 # acceleration exponent [-]
    T::Float64 # desired time headway [s]
    v_des::Float64 # desired speed [m/s]
    s_min::Float64 # minimum acceptable gap [m]
    a_max::Float64 # maximum acceleration ability [m/s²]
    d_cmf::Float64 # comfortable deceleration [m/s²] (positive)
    d_max::Float64 # maximum decelleration [m/s²] (positive)

    t_d::Float64 # reaction time (time delay in responding) [s]
    buf::CircularBuffer{Tuple{Float64, Float64}} # state buffer
 
    function DelayedIntelligentDriverModel(Δt;
        σ::Float64     =   NaN,
        k_spd::Float64 =   1.0,
        δ::Float64     =   4.0,
        T::Float64     =   1.5,
        v_des::Float64 =  29.0, # typically overwritten
        s_min::Float64 =   5.0,
        a_max::Float64 =   3.0,
        d_cmf::Float64 =   2.0,
        d_max::Float64 =   9.0,
        t_d::Float64   =   0.0,
        )

        retval = new()
        retval.a = NaN
        retval.k_spd = k_spd
        retval.δ     = δ
        retval.T     = T
        retval.v_des = v_des
        retval.s_min = s_min
        retval.a_max = a_max
        retval.d_cmf = d_cmf
        retval.d_max = d_max

        # delayed reaction members
        retval.t_d = t_d
        # buffer holds states from previous timesteps
        # where state = (s_gap, lead_velocity)
        buf_size = Int(ceil(t_d / Δt)) + 1
        buf = CircularBuffer{Tuple{Float64, Float64}}(buf_size)

        # start the buffer filled with states that will cause the vehicle to 
        # act as though it does not have a lead vehicle
        append!(buf, fill!(Vector{Tuple{Float64,Float64}}(buf_size), (-Inf, 0)))
        retval.buf = buf

        retval
    end
end

get_name(::DelayedIntelligentDriverModel) = "Delayed_IDM"

function set_desired_speed!(model::DelayedIntelligentDriverModel, 
        v_des::Float64)
    model.v_des = v_des
    model
end

function track_longitudinal!(model::DelayedIntelligentDriverModel, 
        scene::Scene, roadway::Roadway, ego_index::Int, target_index::Int)
    
    # get the ego vehicle and its velocity
    veh_ego = scene[ego_index]
    v = veh_ego.state.v

    # first add samples to the buffer so that t_d = 0 gives the current sample
    # if the target exists at the current timestep, then add its state to the buffer
    if target_index > 0
        veh_target = scene[target_index]
        s_gap = get_frenet_relative_position(get_rear_center(veh_target),
                                            veh_ego.state.posF.roadind, roadway).Δs
        push!(model.buf, (s_gap, veh_target.state.v))

    # if a target is not found, then add (s_gap, lead_velocity) value s.t. 
    # driver behaves as though no vehicle is in front (i.e., s_gap = -Inf)
    else
        push!(model.buf, (-Inf, 0))
    end

    # regardless of whether a target vehicle _currently_ exists, act according
    # to the stored state in the buffer
    s_gap, lead_v = model.buf[1]

    # s_gap > 0.0, then execute IDM model
    if s_gap > 0.0
        Δv = lead_v - v
        s_des = model.s_min + v*model.T - v*Δv / (2*sqrt(model.a_max*model.d_cmf))
        v_ratio = model.v_des > 0.0 ? (v/model.v_des) : 1.0
        model.a = model.a_max * (1.0 - v_ratio^model.δ - (s_des / s_gap)^2)
    
    # if collision imminent, apply maximum decel 
    elseif s_gap > -veh_ego.def.length
        model.a = -model.d_max

    # otherwise, drive to match desired velocity
    else
        Δv = model.v_des - v
        model.a = Δv*model.k_spd
    end

    low = v < 0. ? 0. : -model.d_max
    model.a = clamp(model.a, low, model.a_max)
    model
end

function observe!(model::DelayedIntelligentDriverModel, 
        scene::Scene, roadway::Roadway, egoid::Int)    
    # update the predicted accel
    vehicle_index = get_index_of_first_vehicle_with_id(scene, egoid)
    veh_ego = scene[vehicle_index]
    fore_res = get_neighbor_fore_along_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())

    track_longitudinal!(model, scene, roadway, vehicle_index, fore_res.ind)
    model
end

function Base.rand(model::DelayedIntelligentDriverModel)
    if isnan(model.σ) || model.σ ≤ 0.0
        model.a
    else
        rand(Normal(model.a, model.σ))
    end
end

function Distributions.pdf(model::DelayedIntelligentDriverModel, a_lon::Float64)
    if isnan(model.σ) || model.σ ≤ 0.0
        Inf
    else
        pdf(Normal(model.a, model.σ), a_lon)
    end
end

function Distributions.logpdf(model::DelayedIntelligentDriverModel, a_lon::Float64)
    if isnan(model.σ) || model.σ ≤ 0.0
        Inf
    else
        logpdf(Normal(model.a, model.σ), a_lon)
    end
end