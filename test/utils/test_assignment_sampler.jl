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

function test_swap_discretization()
    var_edges = Dict(:vel=>[-1., 0., 1., 2., 3.], :pos=>[-1.,0.,1.])
    src = UniformAssignmentSampler(var_edges)

    var_edges = Dict(:vel=>[0., 1., 2.], :pos=>[-2.,-1.,0.,1.,2.])
    dest = UniformAssignmentSampler(var_edges)

    a = Assignment()
    a[:vel] = 1
    dest_a = swap_discretization(a, src, dest)
    @test dest_a[:vel] == 1

    a[:vel] = 2
    dest_a = swap_discretization(a, src, dest)
    @test dest_a[:vel] == 1

    a[:vel] = 4
    dest_a = swap_discretization(a, src, dest)
    @test dest_a[:vel] == 2

    a = Assignment()
    a[:pos] = 1
    dest_a = swap_discretization(a, src, dest)
    @test dest_a[:pos] == 2

    a[:pos] = 2
    dest_a = swap_discretization(a, src, dest)
    @test dest_a[:pos] == 3
end

@time test_uniform_assignment_sampler()
@time test_swap_discretization()