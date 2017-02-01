
push!(LOAD_PATH, "../")
include("../testing_utils.jl")

function test_fprop_network_performance()
    input_dim = 32
    hidden_dim = 128
    output_dim = 8
    num_layers = 3
    net = get_debug_network(input_dim, hidden_dim, output_dim, num_layers)
    
    input = ones(1, input_dim)

    num_runs = 100000
    for x in 1:num_runs
        fprop_network(net, input)
    end
    
end

function main()
    test_fprop_network_performance()
end

@time main()