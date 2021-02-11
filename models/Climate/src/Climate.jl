module Climate

export asvec, co2_data, temp_data, co2sim, tempsim

using Omega
using Plots

include("data.jl")
include("sim.jl")
include("intromodel.jl")
include("plots.jl")

end
