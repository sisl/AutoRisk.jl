# using Base.Test
# using AutoRisk

function test_assignment_sampler()
    discs = Dict{Symbol,AbstractDiscretizer}(:vel=>LinearDiscretizer([0., 1., 2.]), 
        :pos=>LinearDiscretizer([-1.,0.,1.]))
    samp = AssignmentSampler(discs)
    a = Assignment(:vel=>1, :pos=>2)
    values = rand(samp, a)
    @test 0. ≤ values[:vel] ≤ 1.
    @test 0. ≤ values[:pos] ≤ 1.

    a = Assignment(:vel=>2, :pos=>1)
    values = rand(samp, a)
    @test 1. ≤ values[:vel] ≤ 2.
    @test -1. ≤ values[:pos] ≤ 0.
end

function test_swap_discretization()
    discs = Dict{Symbol,AbstractDiscretizer}(:vel=>LinearDiscretizer([-1., 0., 1., 2., 3.]), 
        :pos=>LinearDiscretizer([-1.,0.,1.]))
    src = AssignmentSampler(discs)

    discs = Dict{Symbol,AbstractDiscretizer}(:vel=>LinearDiscretizer([0., 1., 2.]), 
        :pos=>LinearDiscretizer([-2.,-1.,0.,1.,2.]))
    dest = AssignmentSampler(discs)

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

function test_categorical_disc_type()
    discs = Dict{Symbol,AbstractDiscretizer}(
        :vel=>LinearDiscretizer([-1., 0., 1., 2., 3.]), 
        :pos=>CategoricalDiscretizer([1.,2.]))
    samp = AssignmentSampler(discs)
    a = Assignment(:vel=>2, :pos=>2)
    v = rand(samp, a)
    @test v[:pos] == 2
end

@time test_assignment_sampler()
@time test_swap_discretization()
@time test_categorical_disc_type()