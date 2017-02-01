# using Base.Test
# using AutoRisk

# push!(LOAD_PATH, "../../src/utils/")
# include("../../src/utils/automotive.jl")

function test_inverse_ttc_to_ttc()
    # missing
    inv_ttc = FeatureValue(0.0, FeatureState.MISSING)
    ttc = inverse_ttc_to_ttc(inv_ttc)
    @test ttc.i == FeatureState.MISSING
    @test ttc.v == 0.0

    # pulling away
    inv_ttc = FeatureValue(0.0, FeatureState.GOOD)
    ttc = inverse_ttc_to_ttc(inv_ttc, censor_hi = 30.)
    @test ttc.i == FeatureState.CENSORED_HI
    @test ttc.v == 30.0

    # collision
    value = 10.0
    inv_ttc = FeatureValue(value, FeatureState.CENSORED_HI)
    ttc = inverse_ttc_to_ttc(inv_ttc)
    @test ttc.i == FeatureState.GOOD
    @test ttc.v == 1. / value
end

@time test_inverse_ttc_to_ttc()