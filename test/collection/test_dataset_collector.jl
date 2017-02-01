# using Base.Test
# using AutoRisk

function test_reset_determinism()
    col_1 = build_debug_dataset_collector(
        output_filepath = "../data/test_dataset_collector_1.h5")
    reset!(col_1, 3)

    col_2 = build_debug_dataset_collector(
        output_filepath = "../data/test_dataset_collector_2.h5")
    reset!(col_2, 3)

    @test col_1.roadway == col_2.roadway
    @test col_1.scene == col_2.scene

    rm("../data/test_dataset_collector_1.h5")
    rm("../data/test_dataset_collector_2.h5")
end

function test_generate_dataset_determinism()
    filepath = "../data/test_dataset_collector.h5"
    feature_dim, target_dim = NUM_FEATURES, NUM_TARGETS
    col = build_debug_dataset_collector(
        output_filepath = filepath,
        num_samples = 10,
        feature_dim = feature_dim,
        target_dim = target_dim)
    generate_dataset(col)

    file = h5open(filepath, "r")
    features_1 = read(file["risk/features"])
    targets_1 = read(file["risk/targets"])

    @test size(features_1, 1) == feature_dim
    @test size(targets_1, 1) == target_dim

    rm(filepath)

    col = build_debug_dataset_collector(
        output_filepath = filepath,
        num_samples = 10,
        feature_dim = feature_dim,
        target_dim = target_dim)
    generate_dataset(col)

    file = h5open(filepath, "r")
    features_2 = read(file["risk/features"])
    targets_2 = read(file["risk/targets"])

    @test features_1 == features_2
    @test targets_1 == targets_2

    rm(filepath)
end

function test_generate_dataset()
    filepath = "../data/test_dataset_collector.h5"
    col = build_debug_dataset_collector(
        output_filepath = filepath,
        num_samples = 2,
        min_num_veh = 10,
        max_num_veh = 10,
        chunk_dim = 1,
    )
    generate_dataset(col)
    file = h5open(filepath, "r")
    features = read(file["risk/features"])
    targets = read(file["risk/targets"])
    
    @test !any(isnan(features))
    @test !any(isnan(targets))

    rm(filepath)
end


@time test_reset_determinism()
@time test_generate_dataset_determinism()
@time test_generate_dataset()
