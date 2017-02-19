export
    Dataset,
    update!,
    finalize!,
    aggregate_datasets

"""
# Description:
    - Dataset type is used as an abstraction over an HDF5 file
"""
type Dataset
    filepath::String
    file::HDF5File

    feature_dim::Int64
    feature_timesteps::Int64
    features::HDF5Dataset

    target_dim::Int64
    targets::HDF5Dataset

    chunk_dim::Int64
    max_num_samples::Int64

    next_idx::Int64
    seeds::Vector{Int64}
    batch_idxs::Vector{Int64}

    """
    # Args:
        - filepath: filepath where to create HDF5 file
        - feature_dim: dimension of features
        - target_dim: dimension of targets
        - max_num_samples: maximum possible samples over course of 
            dataset gathering
        - chunk_dim: the dimension of the hdf5 chunk write sizes
        - init_file: whether to initialize the hdf5 file at construction
            set to false for parallel dataset gathering
    """
    function Dataset(filepath::String, feature_dim::Int64, 
            feature_timesteps::Int64, target_dim::Int64,
            max_num_samples::Int64; chunk_dim::Int64 = 100, 
            init_file::Bool = true)
        retval = new()
        retval.filepath = filepath
        retval.feature_dim = feature_dim
        retval.feature_timesteps = feature_timesteps
        retval.target_dim = target_dim
        retval.chunk_dim = chunk_dim
        retval.max_num_samples = max_num_samples
        retval.next_idx = 1
        retval.seeds = Vector{Int64}(0)
        retval.batch_idxs = Vector{Int64}(0)

        # conditionally create h5 file in constructor
        # note that in the parallel case, creating the file here
        # would require each individual process to reopen the file, 
        # so delay file creation to update! in that case
        if init_file
            initialize!(retval)
        end

        return retval
    end
end

"""
# Description:
    - initialize the hdf5 file underlying this dataset. This should
        be called the first time in update! for parallel dataset gathering.

# Args:
    - dataset: the dataset for which to initialize the hdf5 file
"""
function initialize!(dataset::Dataset)
    # create the file and sub-datasets
    h5file = h5open(dataset.filepath, "w")
    risk_dataset = g_create(h5file, "risk")
    features = d_create(risk_dataset, "features", datatype(Float64), 
                dataspace(dataset.feature_dim, dataset.feature_timesteps, 
                dataset.max_num_samples), 
                "chunk", (dataset.feature_dim, dataset.feature_timesteps, dataset.chunk_dim))
    targets = d_create(risk_dataset, "targets", datatype(Float64), 
                dataspace(dataset.target_dim, dataset.max_num_samples), 
                "chunk", (dataset.target_dim, dataset.chunk_dim))

    # set the values in the dataset for easier access later
    dataset.file = h5file
    dataset.features = dataset.file["risk/features"]
    dataset.targets = dataset.file["risk/targets"]
end

"""
# Description:
    - update the dataset with the given batch of features and targets

# Args:
    - dataset: dataset to update
    - features: features to add, shape = (feature_dim, num_samples)
    - targets: targets to add, shape = (target_dim, num_samples)
    - seed: the random seed used to generate these features and targets
"""
function update!(dataset::Dataset, features::Array{Float64}, 
        targets::Array{Float64}, seed::Int64)
    
    # check for delayed initialization of the h5 file 
    # and initialize if not yet created (should occur
    # in the parallel case)
    if !isdefined(dataset, :file)
        initialize!(dataset)
    end
    
    # the number of samples added each time may differ, so extract here
    num_samples = size(features, 3)
    s = dataset.next_idx
    e = dataset.next_idx + num_samples - 1

    # update features and targets
    dataset.features[:, :, s:e] = features
    dataset.targets[:, s:e] = targets
    dataset.next_idx = e + 1

    # batch_idxs are where batches end
    push!(dataset.batch_idxs, e)
    push!(dataset.seeds, seed)
end

"""
# Description:
    - Finalize the dataset collection process by closing the file and 
        setting the final size of the sub-datasets. This is necessary
        because we do not know beforehand how many samples will be 
        added to the dataset.

# Args:
    - dataset: dataset to finalize
"""
function finalize!(dataset::Dataset)
    # reduce feature and target size
    set_dims!(dataset.features, (dataset.feature_dim, dataset.feature_timesteps, 
        dataset.next_idx - 1))
    set_dims!(dataset.targets, (dataset.target_dim, dataset.next_idx - 1))
    
    # add seeds and batch idxs only at the end
    d_write(dataset.file, "risk/batch_idxs", dataset.batch_idxs)
    d_write(dataset.file, "risk/seeds", dataset.seeds)

    close(dataset.file)
end

"""
# Description:
    - Function for aggregating a set of datasets into a single 
        dataset. Each dataset must have "feature", "target", "seeds",
        and "batch_idxs" datasets. Note that aggregating hdf5 files 
        may not be necessary since you can reference external datasets 
        from a single hdf5 dataset. This is just a convenience
        to only deal with a single file.

# Args:
    - input_filepaths: filepaths to datasets to aggregate
    - output_filepath: filepath to save aggregate dataset
"""
function aggregate_datasets(input_filepaths::Vector{String}, 
        output_filepath::String)

    # compute aggregate size of dataset and feature and target 
    # size by iterating through the input files first
    num_features, num_targets, feature_timesteps = -1, -1, -1
    num_samples = 0

    for (idx, filepath) in enumerate(input_filepaths)
        h5open(filepath, "r") do proc_file
            if idx == 1
                num_features = size(proc_file["risk/features"], 1)
                feature_timesteps = size(proc_file["risk/features"], 2)
                num_targets = size(proc_file["risk/targets"], 1)
            end

            num_proc_features, num_proc_timesteps, num_proc_samples = size(
                proc_file["risk/features"])
            num_proc_targets = size(proc_file["risk/targets"], 1)

            # check that feature and target dims match across sets
            if num_proc_features != num_features
                throw(ErrorException("not all proc datasets have 
                    the same target dim. This proc: $(num_proc_features)
                    previous procs: $(num_features). Filepath: $(filepath)"))
            end
            if num_proc_targets != num_targets
                throw(ErrorException("not all proc datasets have 
                    the same target dim. This proc: $(num_proc_targets)
                    previous procs: $(num_targets). Filepath: $(filepath)"))
            end

            num_samples += num_proc_samples
        end
    end

    # set up aggregate dataset containers
    h5file = h5open(output_filepath, "w")
    risk_dataset = g_create(h5file, "risk")
    feature_set = d_create(risk_dataset, "features", 
        datatype(Float64), dataspace(num_features, feature_timesteps, num_samples))
    target_set = d_create(risk_dataset, "targets", 
        datatype(Float64), dataspace(num_targets, num_samples))
    seeds = Vector{Int64}()
    batch_idxs = Vector{Int64}()

    # collect across datasets
    sidx = 0
    for filepath in input_filepaths
        # collect features and targets
        proc_file = h5open(filepath, "r")
        num_proc_samples = size(proc_file["risk/features"], 3)
        eidx = sidx + num_proc_samples
        feature_set[:, :, sidx + 1:eidx] = read(proc_file["risk/features"])
        target_set[:, sidx + 1:eidx] = read(proc_file["risk/targets"])
        sidx += num_proc_samples

        # collect seeds and batch_idxs
        append!(seeds, read(proc_file["risk/seeds"]))
        cur_batch_idxs = read(proc_file["risk/batch_idxs"])
        if length(batch_idxs) != 0
            last_batch_idx = batch_idxs[end]
            cur_batch_idxs += last_batch_idx
        end
        append!(batch_idxs, cur_batch_idxs)
    end

    # close file 
    close(h5file)

    # write seeds and batch_idxs
    h5write(output_filepath, "risk/seeds", seeds)
    h5write(output_filepath, "risk/batch_idxs", batch_idxs)

end