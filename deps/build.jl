urls = [
    "https://github.com/tawheeler/Vec.jl.git",
    "https://github.com/tawheeler/AutomotiveDrivingModels.jl.git",
    "https://github.com/tawheeler/AutoViz.jl.git",
    "https://github.com/tawheeler/ForwardNets.jl.git"
]

for url in urls
    try
        Pkg.clone(url)
    catch e
        println("Exception when cloning $(url): $(e)")  
    end
end

Pkg.build("AutomotiveDrivingModels")
Pkg.build("AutoRisk")
