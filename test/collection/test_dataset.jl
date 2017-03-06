# using Base.Test
# using AutoRisk

function test_update()
    output_filepath = "data/test_dataset.h5"
    feature_dim = 1
    feature_timesteps = 1
    target_dim = 1
    max_num_samples = 2
    dataset = Dataset(output_filepath, feature_dim, feature_timesteps, target_dim, 
        max_num_samples, chunk_dim = 1)
    update!(dataset, ones(1,1,1), [2.], 3)
    update!(dataset, ones(1,1,1)*4, [5.], 6)
    expected = zeros(1,1,2)
    expected[:,:,1] = 1
    expected[:,:,2] = 4
    @test read(dataset.features) == expected
    @test read(dataset.targets) == [2. 5.]
    @test dataset.seeds == [3, 6]

    # remove filepath after each test
    rm(output_filepath)
end

function test_dataset()
    output_filepath = "data/test_dataset.h5"
    feature_dim = 4
    target_dim = 3
    feature_timesteps = 1
    max_num_samples = 4
    dataset = Dataset(output_filepath, feature_dim, feature_timesteps, target_dim, max_num_samples,
        chunk_dim = 2)

    half_max = Int(max_num_samples / 2.)
    fs = ones(feature_dim, 1, half_max) * 2.
    ts = ones(target_dim, half_max) * 3.
    update!(dataset, fs, ts, 1)
    update!(dataset, fs[:, 1:1, 1:end - 1], ts[:, 1:end - 1], 2)

    # check before finalize
    @test dataset.seeds == [1, 2]
    @test dataset.batch_idxs == [half_max, max_num_samples - 1]
    expected_features = ones(feature_dim, feature_timesteps, max_num_samples) * 2.
    expected_features[:, :, end] = 0
    @test read(dataset.features) == expected_features
    expected_targets = ones(target_dim, max_num_samples) * 3.
    expected_targets[:, end] = 0 
    @test read(dataset.targets) == expected_targets

    finalize!(dataset)
    h5open(output_filepath, "r") do file
        @test size(read(file, "risk/features")) == (feature_dim, feature_timesteps, max_num_samples - 1)
        @test size(read(file, "risk/targets")) == (target_dim, max_num_samples - 1)
        @test read(file, "risk/seeds") == [1, 2]
        @test read(file, "risk/batch_idxs") == [half_max, max_num_samples - 1]
    end

    # remove filepath after each test
    rm(output_filepath)
end

@time test_update()
@time test_dataset()