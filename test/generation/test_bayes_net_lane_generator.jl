# using Base.Test
# using AutoRisk

function test_bayes_net_lane_gen_sampling()
    num_samples = 1000
    num_vars = 5
    data = ones(Int, num_vars, num_samples) * 2
    data[:,1] = 1
    training_data = DataFrame(
            velocity = data[1,:],
            forevelocity = data[2,:],
            foredistance = data[3,:],
            aggressiveness = data[4,:],
            isattentive = data[5,:]
    )
    base_bn = fit(DiscreteBayesNet, training_data, (
            :isattentive=>:foredistance, 
            :isattentive=>:velocity,
            :aggressiveness=>:foredistance, 
            :aggressiveness=>:velocity,
            :foredistance=>:velocity,
            :forevelocity=>:velocity
        )
    )
    new_values = ones(Int, num_samples)
    new_values[1] = 2
    training_data[:isattentive] = new_values
    prop_bn = fit(DiscreteBayesNet, training_data, (
            :isattentive=>:foredistance, 
            :isattentive=>:velocity,
            :aggressiveness=>:foredistance, 
            :aggressiveness=>:velocity,
            :foredistance=>:velocity,
            :forevelocity=>:velocity
        )
    )
    var_edges = Dict(
        :aggressiveness=>[0.,.5,1.], 
        :foredistance=>[0.,10.,20.],
        :forevelocity=>[0.,5.,10.],
        :velocity=>[0.,5.,10.]
    )
    sampler = UniformAssignmentSampler(var_edges)
    dynamics = Dict(:velocity=>:forevelocity)
    num_veh_per_lane = 2
    min_p = get_passive_behavior_params(err_p_a_to_i = .5)
    max_p = get_aggressive_behavior_params(err_p_a_to_i = .5)
    context = IntegratedContinuous(.1, 1)
    behgen = CorrelatedBehaviorGenerator(context, min_p, max_p)
    gen = BayesNetLaneGenerator(base_bn, prop_bn, sampler, dynamics, num_veh_per_lane, 
        behgen)
    roadway = gen_straight_roadway(1)
    scene = Scene(num_veh_per_lane)
    models = Dict{Int,DriverModel}()
    rand!(gen, roadway, scene, models, 1)

    @test get_weights(gen)[1] < 1.
    @test models[1].is_attentive == false
    @test get_weights(gen)[2] ≈ 1.
    @test models[2].is_attentive == true
end

@time test_bayes_net_lane_gen_sampling()
