# using Base.Test
# using AutoRisk

function test_uniform_assignment_sampler()
    var_edges = Dict(:vel=>[0., 1., 2.], :pos=>[-1.,0.,1.])
    samp = UniformAssignmentSampler(var_edges)
    a = Assignment(:vel=>1, :pos=>2)
    values = rand(samp, a)
    @test 0. ≤ values[:vel] ≤ 1.
    @test 0. ≤ values[:pos] ≤ 1.

    a = Assignment(:vel=>2, :pos=>1, :unsampled=>0.)
    values = rand(samp, a)
    @test 1. ≤ values[:vel] ≤ 2.
    @test -1. ≤ values[:pos] ≤ 0.
    @test values[:unsampled] == 0.
end

@time test_uniform_assignment_sampler()