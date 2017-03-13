export 
    DatasetCollector,
    ParallelDatasetCollector,
    rand!,
    generate_dataset

"""
# Description:
    - DatasetCollector orchestrates the serial collection of a dataset.
"""
type DatasetCollector
    seeds::Vector{Int64}

    roadway_gen::RoadwayGenerator
    scene_gen::SceneGenerator
    behavior_gen::BehaviorGenerator
    eval::Evaluator
    dataset::Dataset

    scene::Scene
    models::Dict{Int, DriverModel}
    roadway::Roadway

    id::Int
    monitor::Any
    function DatasetCollector(seeds::Vector{Int64}, roadway_gen::RoadwayGenerator,
            scene_gen::SceneGenerator, behavior_gen::BehaviorGenerator,
            eval::Evaluator, dataset::Dataset, scene::Scene, 
            models::Dict{Int, DriverModel}, roadway::Roadway; id::Int = 0,
            monitor::Any = Monitor())
        return new(seeds, roadway_gen, scene_gen, behavior_gen, eval, dataset, 
            scene, models, roadway, id, monitor)
    end
end

"""
# Description:
    - Reset the state randomly according to the random seed

# Args:
    - col: the collector being used
    - seed: the random seed uniquely identifying the resulting state
"""
function Base.rand!(col::DatasetCollector, seed::Int64)
    info("id $(col.id) collecting seed $(seed)")
    rand!(col.roadway_gen, col.roadway, seed)
    rand!(col.scene_gen, col.scene, col.roadway, seed)
    rand!(col.behavior_gen, col.models, col.scene, seed)
end

"""
# Description:
    - Generate a dataset for each seed of the collector

# Args:
    - col: the collector to use
"""
function generate_dataset(col::DatasetCollector)
    for seed in col.seeds
        rand!(col, seed)
        evaluate!(col.eval, col.scene, col.models, col.roadway, seed)
        update!(col.dataset, get_features(col.eval), get_targets(col.eval), seed)
        monitor(col.monitor, col, seed)
    end
    finalize!(col.dataset)
end

"""
# Description:
    - ParallelDatasetCollector orchestrates the parallel generation 
        of a dataset.
"""
type ParallelDatasetCollector
    cols::Vector{DatasetCollector}
    output_filepath::String

    """
    # Args:
        - cols: a vector of dataset collectors 
        - seeds: the seeds for which states should be generated and simulated.
            note that these are partitioned in the constructor
        - output_filepath: filepath for the final dataset
    """
    function ParallelDatasetCollector(cols::Vector{DatasetCollector}, 
            seeds::Vector{Int64}, output_filepath::String)
        seedsets = ordered_partition(seeds, length(cols))
        for (col, seeds) in zip(cols, seedsets)
            col.seeds = seeds
        end
        return new(cols, output_filepath)
    end
end

"""
# Description:
    - Generate a dataset in parallel.

# Args:
    - pcol: the parallel dataset collector to use
"""
function generate_dataset(pcol::ParallelDatasetCollector)
    pmap(generate_dataset, pcol.cols)
    filepaths = [c.dataset.filepath for c in pcol.cols]
    aggregate_datasets(filepaths, pcol.output_filepath)
end

