# using Base.Test
# using AutoRisk

# BASE_TEST_DIR = ".."

function test_update()
    output_filepath = BASE_TEST_DIR * "/data/test_dataset.h5"
    feature_dim = 1
    feature_timesteps = 1
    target_dim = 1
    target_timesteps = 2
    max_num_samples = 2
    dataset = Dataset(
        output_filepath, 
        feature_dim, 
        feature_timesteps, 
        target_dim, 
        target_timesteps,
        max_num_samples, 
        chunk_dim = 1
    )
    update!(dataset, ones(1,1,1), ones(1,2,1) * 2, 3)
    update!(dataset, ones(1,1,1)*4, ones(1,2,1) * 5, 6)
    expected = zeros(1,1,2)
    expected[:,:,1] .= 1
    expected[:,:,2] .= 4
    @test read(dataset.features) == expected
    expected = zeros(1,2,2)
    expected[:,:,1] .= 2
    expected[:,:,2] .= 5
    @test read(dataset.targets) == expected
    @test dataset.seeds == [3, 6]

    # remove filepath after each test
    rm(output_filepath)
end

function test_dataset()
    output_filepath = BASE_TEST_DIR * "/data/test_dataset.h5"
    feature_dim = 4
    target_dim = 3
    feature_timesteps = 1
    target_timesteps = 2
    max_num_samples = 4
    dataset = Dataset(
        output_filepath, 
        feature_dim, 
        feature_timesteps, 
        target_dim, 
        target_timesteps,
        max_num_samples,
        chunk_dim = 2)

    half_max = Int(max_num_samples / 2.)
    fs = ones(feature_dim, 1, half_max) * 2.
    ts = ones(target_dim, 2, half_max) * 3.
    update!(dataset, fs, ts, 1)
    update!(dataset, fs[:, :, 1:end - 1], ts[:, :, 1:end - 1], 2)

    # check before finalize
    @test dataset.seeds == [1, 2]
    @test dataset.batch_idxs == [half_max, max_num_samples - 1]
    expected_features = ones(feature_dim, feature_timesteps, max_num_samples) * 2.
    expected_features[:, :, end] .= 0
    @test read(dataset.features) == expected_features
    expected_targets = ones(target_dim, target_timesteps, max_num_samples) * 3.
    expected_targets[:, :, end] .= 0 
    @test read(dataset.targets) == expected_targets

    finalize!(dataset)
    h5open(output_filepath, "r") do file
        @test size(read(file, "risk/features")) == (feature_dim, feature_timesteps, max_num_samples - 1)
        @test size(read(file, "risk/targets")) == (target_dim, target_timesteps, max_num_samples - 1)
        @test read(file, "risk/seeds") == [1, 2]
        @test read(file, "risk/batch_idxs") == [half_max, max_num_samples - 1]
    end

    # remove filepath after each test
    rm(output_filepath)
end

function test_weighted_dataset()
    output_filepath = BASE_TEST_DIR * "/data/test_weight_dataset.h5"
    feature_dim = 4
    target_dim = 3
    feature_timesteps = 1
    target_timesteps = 2
    max_num_samples = 4
    dataset = Dataset(
        output_filepath, 
        feature_dim, 
        feature_timesteps,
        target_dim, 
        target_timesteps,
        max_num_samples,
        chunk_dim = 2, 
        use_weights = true)

    half_max = Int(max_num_samples / 2.)
    fs = ones(feature_dim, 1, half_max) * 2.
    ts = ones(target_dim, 2, half_max) * 3.
    ws = ones(1, half_max)
    update!(dataset, fs, ts, ws, 1)
    ws *= 2
    update!(dataset, fs[:, :, 1:end - 1], ts[:, :, 1:end - 1], ws[:, 1:end - 1], 2)

    # check before finalize
    # checking for the fact that multiple updates were collected, and that 
    # after finalizing, the remaining entry that had not been updated is not 
    # saved
    @test dataset.seeds == [1, 2]
    @test dataset.batch_idxs == [half_max, max_num_samples - 1]
    expected_features = ones(feature_dim, feature_timesteps, max_num_samples) * 2.
    expected_features[:, :, end] .= 0
    @test read(dataset.features) == expected_features
    expected_targets = ones(target_dim, target_timesteps, max_num_samples) * 3.
    expected_targets[:, :, end] .= 0 
    @test read(dataset.targets) == expected_targets
    expected_weights = ones(1, max_num_samples)
    expected_weights[half_max + 1:end - 1] .= 2
    expected_weights[:, end] .= 0
    @test read(dataset.weights) == expected_weights

    finalize!(dataset)
    h5open(output_filepath, "r") do file
        @test size(read(file, "risk/features")) == (feature_dim, feature_timesteps, max_num_samples - 1)
        @test size(read(file, "risk/targets")) == (target_dim, target_timesteps, max_num_samples - 1)
        @test size(read(file, "risk/weights")) == (1, max_num_samples - 1)
        @test read(file, "risk/seeds") == [1, 2]
        @test read(file, "risk/batch_idxs") == [half_max, max_num_samples - 1]
    end

    # remove filepath after each test
    rm(output_filepath)
end

@time test_update()
@time test_dataset()
@time test_weighted_dataset()