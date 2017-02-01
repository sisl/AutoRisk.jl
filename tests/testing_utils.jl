
# retrieve a debug, single-hidden-layer network
function get_debug_network(input_dim, hidden_dim, output_dim, num_layers=1)
    net = Network()

    # hidden layers
    prev_dim = input_dim
    for layer in 1:num_layers
        push!(net.weights, ones(prev_dim, hidden_dim))
        push!(net.biases, ones(1, hidden_dim))
        prev_dim = hidden_dim
    end

    # output layer
    push!(net.weights, ones(hidden_dim, output_dim))
    push!(net.biases, ones(1, output_dim))

    return net
end

function sigmoid(values::Array{Float64})
    return ones(values) ./ (ones(values) + exp(-values))
end

function sigmoid(value::Float64)
    return 1 / (1 + exp(-value))
end