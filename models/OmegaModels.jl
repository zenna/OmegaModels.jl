__precompile__(false)
"Collection of Models"
module OmegaModels

using Omega
using Tensorboard
using FileIO
using ZenUtils

include("common.jl")

include("invgraphics/invgraphics.jl")
include("mnist/mnist.jl")
include("programlearn/programlearn.jl")
# include("rcd/rcd.jl")
include("rnn/rnn.jl")
include("small/small.jl")
include("spelke/spelke.jl")
include("thermostat/thermostat.jl")

include("fairness.jl")
include("linear.jl")
include("robustness.jl")

end