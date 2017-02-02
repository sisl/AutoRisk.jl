try
    Pkg.clone("https://github.com/tawheeler/Vec.jl.git")
catch e
    println("Exception when cloning Vec.jl while building AutoRisk: $(e)")
end
try
    Pkg.clone("https://github.com/tawheeler/AutomotiveDrivingModels.jl.git")
catch e
    println("Exception when cloning AutomotiveDrivingModels.jl while building AutoRisk: $(e)")	
end
try
    Pkg.clone("https://github.com/tawheeler/AutoViz.jl.git")
catch e
    println("Exception when cloning AutoViz.jl while building AutoRisk: $(e)")	
end