
using HDF5

# wraps weights and biases
type Network
    weights::Vector{Array{Float64}}
    biases::Vector{Array{Float64}}
    nonlinearity::Function
    function Network()
        return new([], [], relu!)
    end
end

#=
Description:
    - Loads a network object from a hdf5 weights file. 

Args:
    - weights_filepath: filepath to load network from

Returns:
    - network: network object with weights and biases from file
=#
function load_network(weights_filepath)
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
    
    return network
end

#=
Description
    - Relu nonlinearity, sets everything below zero to one

Args:
    - values: array of values to mutate
=#
function relu!(values::Array{Float64})
    values[values .< 0] = 0
end

#=
Description
    - Pass the input values through the sigmoid operation

Args:
    - values: array of values to mutate
=#
function sigmoid(values::Array{Float64})
    return ones(values) ./ (ones(values) + exp(-values))
end

#=
Description:
    - Forward propagate input through a network.

Args:
    - network: network to fprop through
    - input: value to fprop, shape (batch_size, input_dim)
    note that the individual inputs should be row not column vectors.

Returns:
    - output from the network
=#
function fprop_network(network::Network, input::Array{Float64})
    # fprop through network
    num_layers = length(network.weights)
    state = input
    for lidx in 1:num_layers
        state = state * network.weights[lidx] + network.biases[lidx]
        if lidx != num_layers
            # mutates state
            network.nonlinearity(state)
        end
    end
    return sigmoid(state)
end;

