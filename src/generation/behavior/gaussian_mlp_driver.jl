export
    GaussianMLPDriver,
    get_name,
    action_context,
    reset_hidden_state!,
    observe!

type GaussianMLPDriver{A<:DriveAction, F<:Real, G<:Real, E<:AbstractFeatureExtractor, M<:MvNormal} <: DriverModel{A, ActionContext}
    net::ForwardNet
    rec::SceneRecord
    pass::ForwardPass
    input::Vector{F}
    output::Vector{G}
    extractor::E
    mvnormal::M
    context::ActionContext
    features::Array{Float64}

    a_hi::Float64
    a_lo::Float64
    ω_hi::Float64
    ω_lo::Float64   
end

_get_Σ_type{Σ,μ}(mvnormal::MvNormal{Σ,μ}) = Σ
function GaussianMLPDriver{A <: DriveAction}(::Type{A}, 
        net::ForwardNet, 
        extractor::AbstractFeatureExtractor, 
        context::ActionContext;
        input::Symbol = :input,
        output::Symbol = :output,
        Σ::Union{Real, Vector{Float64}, Matrix{Float64},  Distributions.AbstractPDMat} = 0.1,
        rec::SceneRecord = SceneRecord(2, get_timestep(context)),
        a_hi::Float64 = 3.,
        a_lo::Float64 = -5.,
        ω_hi::Float64 = .01,
        ω_lo::Float64 = -.01)

    pass = calc_forwardpass(net, [input], [output])
    input_vec = net[input].tensor
    output = net[output].tensor
    features = zeros(Float64, (length(extractor), 500))
    mvnormal = MvNormal(Array(Float64, 2), Σ)
    GaussianMLPDriver{A, eltype(input_vec), eltype(output), typeof(extractor), typeof(mvnormal)}(net, rec, pass, input_vec, output, extractor, mvnormal, context, features, a_hi, a_lo, ω_hi, ω_lo)
end
AutomotiveDrivingModels.get_name(::GaussianMLPDriver) = "GaussianMLPDriver"
AutomotiveDrivingModels.action_context(model::GaussianMLPDriver) = model.context

function AutomotiveDrivingModels.reset_hidden_state!(model::GaussianMLPDriver)
    empty!(model.rec)
    model
end

function AutomotiveDrivingModels.observe!{A,F,G,E,P}(
        model::GaussianMLPDriver{A,F,G,E,P},
        scene::Scene, 
        roadway::Roadway, 
        egoid::Int)
    update!(model.rec, scene)
    vehicle_index = get_index_of_first_vehicle_with_id(scene, egoid)
    o = pull_features!(model.extractor, model.features, model.rec, roadway, vehicle_index)
    model.net[:hidden_0].input = o[:,vehicle_index]
    forward!(model.pass)
    copy!(model.mvnormal.μ, model.output[1:2])
    return model
end
function Base.rand{A,F,G,E,P}(model::GaussianMLPDriver{A,F,G,E,P})
    actions = convert(A, rand(model.mvnormal))
    a = min(max(actions.a, model.a_lo), model.a_hi)
    ω = min(max(actions.ω, model.ω_lo), model.ω_hi)
    actions = AccelTurnrate(a, ω)
    return actions
end
Distributions.pdf{A,F,G,E,P}(model::GaussianMLPDriver{A,F,G,E,P}, a::A) = pdf(
    model.mvnormal, convert(Vector{Float64}, a))
Distributions.logpdf{A,F,G,E,P}(model::GaussianMLPDriver{A,F,G,E,P}, a::A) = logpdf(
    model.mvnormal, convert(Vector{Float64}, a))