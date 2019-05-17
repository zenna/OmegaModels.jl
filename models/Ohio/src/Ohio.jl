module Ohio

include("parsedata.jl")
using .ParseData

include("train.jl")
using .Train

include("neuralode.jl")
using .NeuralODE

end