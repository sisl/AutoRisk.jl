export 
    unordered_partition,
    ordered_partition

function unordered_partition(arr::Vector{Int64}, n::Int64)
    return [arr[i:n:end] for i in 1:n]
end

function ordered_partition(arr::Vector{Int64}, n::Int64)
    each = Int(floor(length(arr) / n))
    r = length(arr) % n
    vals = Dict()
    for i in 1:n
        s = (i - 1) * each
        vals[i] = arr[s + 1: s + each]
    end
    append!(vals[n], arr[end - r + 1:end])
    return [vals[i] for i in 1:n]
end