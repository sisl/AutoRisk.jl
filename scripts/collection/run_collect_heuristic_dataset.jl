using AutoRisk

include("collect_heuristic_dataset.jl")
include("heuristic_dataset_config.jl")

function analyze_risk_dataset(output_filepath)
    dataset = h5open(output_filepath)
    features = read(dataset["risk/features"])
    targets = read(dataset["risk/targets"])
    println("avg features: $(mean(features, (3)))")
    println("avg targets: $(mean(targets, (2)))")
    println("size of dataset features: $(size(features))")
    println("size of dataset targets: $(size(targets))")

end

function main()
    parse_flags!(FLAGS, ARGS)
    FLAGS["num_proc"] = max(nprocs() - 1, 1)
    pcol = build_parallel_dataset_collector(FLAGS)
    generate_dataset(pcol)
    analyze_risk_dataset(FLAGS["output_filepath"])
end

@time main()
# Profile.clear_malloc_data()
# Profile.clear()
# @profile main()
# Profile.print(format = :flat, sortedby = :time)