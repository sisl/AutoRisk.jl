# clone
urls = [
    "https://github.com/tawheeler/Vec.jl.git",
    "https://github.com/tawheeler/AutomotiveDrivingModels.jl.git",
    "https://github.com/tawheeler/AutoViz.jl.git",
    "https://github.com/tawheeler/ForwardNets.jl.git",
    "https://github.com/sisl/BayesNets.jl.git",
    "https://github.com/sisl/GridInterpolations.jl.git"
]

packages = keys(Pkg.installed())
for url in urls
    try
        id1 = search(url, "https://github.com/")[end]
        offset = search(url[(id1[end]+1):end], "/")[end]
        package = url[(id1+offset+1): (search(url,".jl.git")[1]-1)]
        if !in(package, packages)
          Pkg.clone(url)
        else
          println("$(package) already exists. Not cloning.")
        end
    catch e
        println("Exception when cloning $(url): $(e)")
    end
end

Pkg.build("AutomotiveDrivingModels")
Pkg.build("BayesNets")

# checkout specific branches
checkouts = [
    ("ForwardNets", "nextgen")
]

for (pkg, branch) in checkouts
    Pkg.checkout(pkg, branch)
end
