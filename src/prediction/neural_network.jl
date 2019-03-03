export 
    PredictionModel,
    Network,
    load_network,
    predict,
    normalize_input!,
    fprop_network

abstract type PredictionModel end

predict(model::PredictionModel, input::Array{Float64}) = error(
    "predict not implemented for $(model)")

# wraps weights and biases
mutable struct Network <: PredictionModel
    weights::Vector{Array{Float64}}
    biases::Vector{Array{Float64}}
    nonlinearity::Function
    means::Array{Float64}
    stds::Array{Float64}
    function Network()
        return new([], [], relu!, [], [])
    end
end

"""
Description:
    - Loads a network object from a hdf5 weights file. 

Args:
    - weights_filepath: filepath to load network from

Returns:
    - network: network object with weights and biases from file
"""
function load_network(weights_filepath::String)
    network = Network()

    # read in weights
    weights = h5open(weights_filepath, "r") do file
        read(file, "weights")
    end

    # each key in weights corresponds to a layer
    # make the assumption that the sorted keys 
    # give the correct order of the layers
    for layer_key in sort(collect(keys(weights)))
        bias_key, weights_key = sort(collect(keys(weights[layer_key])))
        push!(network.weights, transpose(weights[layer_key][weights_key]))
        push!(network.biases, transpose(weights[layer_key][bias_key]))
    end

    # read in the input means and standard deviations
    stats = h5open(weights_filepath, "r") do file
        read(file, "stats")
    end
    input_dim = length(stats["means"])
    network.means = reshape(stats["means"], (1, input_dim))
    network.stds = reshape(stats["stds"], (1, input_dim))
    
    return network
end

function Network(filepath::String)
    return load_network(filepath)
end

"""
Description
    - Relu nonlinearity, sets everything below zero to one

Args:
    - values: array of values to mutate
"""
function relu!(values::Array{Float64})
    values[values .< 0] = 0
end

"""
Description
    - Pass the input values through the sigmoid operation

Args:
    - values: array of values to mutate
"""
function sigmoid(values::Array{Float64})
    return ones(values) ./ (ones(values) + exp(-values))
end

"""
Description:
    - Forward propagate input through a network.

Args:
    - network: network to fprop through
    - input: value to fprop, shape (batch_size, input_dim)
    note that the individual inputs should be row not column vectors.

Returns:
    - output from the network
"""
function fprop_network(network::Network, input::Array{Float64})
    # fprop through network
    num_layers = length(network.weights)
    state = input
    for lidx in 1:num_layers
        state = state * network.weights[lidx] .+ network.biases[lidx]
        if lidx != num_layers
            # mutates state
            network.nonlinearity(state)
        end
    end
    return sigmoid(state)
end

"""
Description:
    - Normalize the input values by mean subtraction and division by 
        standard deviation

Args:
    - network: network storing the means and std devs
    - input: input to normalize
"""
function normalize_input!(network::Network, input::Array{Float64})
    input .-= network.means
    input ./= network.stds
end

"""
Description:
    - Predict values for given input

Args:
    - network: network to use in prediction
    - input: value to fprop, shape (batch_size, input_dim)
    note that the individual inputs should be row not column vectors.

Returns:
    - output from the network
"""
function predict(network::Network, input::Array{Float64})
    normalize_input!(network, input)
    return fprop_network(network, input)
end
