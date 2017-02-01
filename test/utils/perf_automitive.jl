
using AutoRisk

push!(LOAD_PATH, "../../src/utils/")
include("../../src/utils/automotive.jl")

function perf_get_collision_type()

end

@time test_inverse_ttc_to_ttc()