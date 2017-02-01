using Base.Test

const weights_filepath = joinpath(
    dirname(@__FILE__()), "..", "data/networks/test.h5")

function test_load_network()
    network = load_network(weights_filepath)
end

function test_fprop_network() 
    input_dim = 5
    hidden_dim = 10
    output_dim = 3
    net = get_debug_network(input_dim, hidden_dim, output_dim)
    input = ones(1, input_dim)
    output = fprop_network(net, input)
    @test output == ones(1, output_dim) * sigmoid(61.)

    input = ones(1, input_dim) * -1
    output = fprop_network(net, input)
    @test output == sigmoid(ones(1, output_dim))
end

function test_fprop_network_real_weights() 
    input_dim = 1 
    output_dim = 1
    network = load_network(weights_filepath)
    input = ones(1, input_dim)
    output = fprop_network(network, input)
    @test abs(output[1] - 0.00871525) < 1e-4
end

@time test_load_network()
@time test_fprop_network()
@time test_fprop_network_real_weights()
