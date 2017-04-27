# using Base.Test
# using AutoRisk

function test_simple_learning()
    srand(0)
    minpos = 0
    maxpos = 5
    nbins = 6
    bins = linspace(minpos, maxpos, nbins)
    null_bin = linspace(0,2,2) # mocking feature not considered
    grid = RectangleGrid(bins, bins, null_bin)
    target_dim = 2
    predictor = TDPredictor(grid, target_dim, discount = 0.5)

    # generate dataset
    # dataset consists of 3 dimensional states
    # where the last state is irrelevant
    # and the values are noisy functions of x and y
    num_samples = 10000
    eps = 1e-2
    features = zeros(3, num_samples)
    values = zeros(target_dim, num_samples)
    σ = 1
    for i in 1:num_samples
        x = rand(minpos:eps:maxpos)
        y = rand(minpos:eps:maxpos)
        z = rand(-100:100)
        features[:, i] = [x,y,z]
        v1 = 1 
        v2 = σ * randn() + x + y
        values[:, i] = [v1, v2]
    end
    
    # fit the model
    num_epochs = 100
    for epoch in 1:num_epochs
        update!(predictor, features, values)
    end

    # visualize
    num_steps = 10
    steps = linspace(minpos, maxpos, num_steps)
    v = zeros(target_dim, num_steps, num_steps)
    for (i, x) in enumerate(steps)
        for (j, y) in enumerate(steps)
            v[:,i,j] = predict(predictor, [x, y, 0])
        end
    end

    @test all(abs(v[1,:,:] - .5) .< 1e-2)
    @test all(v[2,:,:] .> 0)
    @test v[2,1,1] < v[2,end,end]

    # visualize it
    # function get_heat_1(x, y)
    #     predict(predictor, [x, y, 0])[1]
    # end
    # function get_heat_2(x, y)
    #     predict(predictor, [x, y, 0])[2]
    # end

    # nbins = 50
    # img = Plots.Image(get_heat_1, (minpos, maxpos), (minpos, maxpos), 
    #     xbins=nbins, ybins=nbins)
    # PGFPlots.save("/Users/wulfebw/Desktop/test1.pdf", img)
    # img = Plots.Image(get_heat_2, (minpos, maxpos), (minpos, maxpos), 
    #     xbins=nbins, ybins=nbins)
    # PGFPlots.save("/Users/wulfebw/Desktop/test2.pdf", img)
end

function test_step()
    srand(0)
    minpos = 0
    maxpos = 5
    nbins = 6
    bins = linspace(minpos, maxpos, nbins)
    grid = RectangleGrid(bins, bins)
    target_dim = 2
    predictor = TDPredictor(grid, target_dim, lr = 1., discount = 0.5)

    x = [0.,0.]
    a = [0.]
    r = [1., 1.]
    nx = [0.,0.]
    done = false
    AutoRisk.step(predictor, x, a, r, nx, done)
    @test predict(predictor, x) == r

    AutoRisk.step(predictor, x, a, r, nx, done)
    @test predict(predictor, x) == r + .5 * r
end

@time test_simple_learning()
@time test_step()
