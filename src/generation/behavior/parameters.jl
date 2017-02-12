import Base: rand, ==

export 
    IDMParams,
    MOBILParams,
    LateralParams,
    BehaviorParams,
    uniform,
    get_aggressive_behavior_params,
    get_passive_behavior_params,
    get_normal_behavior_params

function uniform(rng::MersenneTwister, low::Float64, high::Float64)
    return low + rand(rng) * (high - low)
end

type IDMParams
    σ::Float64 # standard deviation of action
    k_spd::Float64 # proportional speed tracking constant
    δ::Float64 # acceleration exponent [-]
    T::Float64 # desired time headway [s]
    v_des::Float64 # desired speed [m/s]
    s_min::Float64 # minimum acceptable gap [m]
    a_max::Float64 # maximum acceleration ability [m/s²]
    d_cmf::Float64 # comfortable deceleration [m/s²] (positive)
    t_d::Float64 # time delay if using delayed response idm [s]
end

function Base.rand(rng::MersenneTwister, min::IDMParams, 
    max::IDMParams)
    return IDMParams(uniform(rng, min.σ, max.σ),
        uniform(rng, min.k_spd, max.k_spd), 
        uniform(rng, min.δ, max.δ), 
        uniform(rng, min.T, max.T),
        uniform(rng, min.v_des, max.v_des), 
        uniform(rng, min.s_min, max.s_min),
        uniform(rng, min.a_max, max.a_max), 
        uniform(rng, min.d_cmf, max.d_cmf),
        uniform(rng, min.t_d, max.t_d))
end

function Base.:(==)(p1::IDMParams, p2::IDMParams)
    return (p1.σ == p2.σ
            && p1.k_spd == p2.k_spd
            && p1.δ == p2.δ
            && p1.T == p2.T
            && p1.v_des == p2.v_des
            && p1.s_min == p2.s_min
            && p1.a_max == p2.a_max
            && p1.d_cmf == p2.d_cmf
            && p1.t_d == p2.t_d)
end

type MOBILParams
    politeness::Float64 # politeness factor
    safe_decel::Float64 # safe braking value
    advantage_threshold::Float64 # minimum accel
end

function Base.rand(rng::MersenneTwister, min::MOBILParams, 
    max::MOBILParams)
    return MOBILParams(uniform(rng, min.politeness, max.politeness), 
        uniform(rng, min.safe_decel, max.safe_decel), 
        uniform(rng, min.advantage_threshold, max.advantage_threshold))
end

function Base.:(==)(p1::MOBILParams, p2::MOBILParams)
    return (p1.politeness == p2.politeness 
            && p1.safe_decel == p2.safe_decel
            && p1.advantage_threshold == p2.advantage_threshold)
end

type LateralParams
    σ::Float64 # standard deviation of action
    kp::Float64 # proportional constant for lane tracking
    kd::Float64 # derivative constant for lane tracking
end

function Base.rand(rng::MersenneTwister, min::LateralParams, 
        max::LateralParams)
    return LateralParams(
        uniform(rng, min.σ, max.σ), 
        uniform(rng, min.kp, max.kp), 
        uniform(rng, min.kd, max.kd))
end

function Base.:(==)(p1::LateralParams, p2::LateralParams)
    return p1.σ == p2.σ && p1.kp == p2.kp && p1.kd == p2.kd
end

type BehaviorParams
    idm::IDMParams
    mobil::MOBILParams
    lat::LateralParams
end

function Base.:(==)(p1::BehaviorParams, p2::BehaviorParams)
    return p1.idm == p2.idm && p1.mobil == p2.mobil && p1.lat == p2.lat
end

function Base.rand(rng::MersenneTwister, min_p::BehaviorParams, 
        max_p::BehaviorParams)
    idm = rand(rng, min_p.idm, max_p.idm)
    mobil = rand(rng, min_p.mobil, max_p.mobil)
    lat = rand(rng, min_p.lat, max_p.lat)
    return BehaviorParams(idm, mobil, lat)
end

# convert params to a vector
function Base.convert(::Type{Vector{Float64}}, p::BehaviorParams)
    values::Vector{Float64} = []
    for f in fieldnames(p)
        fv = getfield(p, f)
        for v in fieldnames(fv)
            push!(values, getfield(fv, v))
        end
    end
    return values
end

""" 
Standard parameter sets
"""

function get_aggressive_behavior_params(;
        lon_σ = 0.0, 
        lat_σ = 0.0, 
        response_time = 0.0)
    return BehaviorParams(
        IDMParams(lon_σ, 1.5, 4.0, 0.5, 35., 4.0, 4.0, 2.5, response_time),
        MOBILParams(0.1, 2.0, 0.01),
        LateralParams(lat_σ, 3.5, 2.5))
end

function get_passive_behavior_params(;
        lon_σ = 0.0, 
        lat_σ = 0.0, 
        response_time = 0.0)
    return BehaviorParams(
        IDMParams(lon_σ, 1.0, 4.0, 1.75, 25., 5.0, 1.0, 1.0, response_time),
        MOBILParams(0.5, 2.0, 0.7),
        LateralParams(lat_σ, 3.0, 2.0))
end

function get_normal_behavior_params(;
        lon_σ = 0.0, 
        lat_σ = 0.0, 
        response_time = 0.0)
    return BehaviorParams(
        IDMParams(lon_σ, 1.25, 4.0, 1.25, 30., 4.5, 2.5, 1.75, response_time),
        MOBILParams(0.3, 2.0, 0.2),
        LateralParams(lat_σ, 3.25, 2.25))
end

# function get_politeness_covariance_matrix(;σ = 1.0, ρ = 0.5, 
#         deterministic = true, delayed_response = true)
#     # negative values indicate values that are smaller for more aggressive 
#     eps = 0.00000001
#     vars = [eps, 0.15, eps, -.35, 3., -.3, 1., .5, .2, 
#             -.1, eps, -.3,
#             eps, .15, .15]

#     num_params = length(vars)
#     covmat = zeros(num_params, num_params)
#     for i in 1:num_params
#         covmat[i,i] = abs(vars[i])
#     end

#     politeness_idx = 10
#     for r in 1:num_params
#         if vars[r] == eps || r == politeness_idx
#             continue
#         end
#         cov = ρ * min(abs(vars[r]), abs(vars[politeness_idx]))
#         if vars[r] > 0.
#             cov *= -1
#         end
#         covmat[r, politeness_idx] = cov
#         covmat[politeness_idx, r] = cov
#     end
# end
