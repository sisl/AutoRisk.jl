using Pkg

# unregistered packages
urls = [
    "https://github.com/sisl/Vec.jl.git",
    "https://github.com/sisl/Records.jl",
    "https://github.com/sisl/AutomotiveDrivingModels.jl.git",
    "https://github.com/sisl/AutoViz.jl.git"
]

packages = keys(Pkg.installed())
for url in urls
    try
  	Pkg.add(PackageSpec(url=url))
    catch e
        println("Exception when cloning $(url): $(e)")
    end
end

Pkg.add(PackageSpec(url="https://github.com/tawheeler/ForwardNets.jl.git",
    rev="nextgen"))
