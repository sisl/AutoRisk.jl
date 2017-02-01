# using Base.Test

# push!(LOAD_PATH, "../../src/utils/")
# include("../../src/utils/utils.jl")

function test_unordered_partition()
    vals = collect(1:7)
    n = 3
    actual = unordered_partition(vals, n)
    expected = [[1,4,7],[2,5],[3,6]]
    @test actual == expected
end

function test_ordered_partition()
    vals = collect(1:7)
    n = 3
    actual = ordered_partition(vals, n)
    expected = [[1,2],[3,4],[5,6,7]]
    @test actual == expected
end

@time test_unordered_partition()
@time test_ordered_partition()