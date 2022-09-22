"Inverse Graphics"
module InvRayTrace

using Omega
using RayTrace
using Flux
import Statistics: mean
using Random
using Callbacks
using Images
using Distributions


import RayTrace: ListScene, rgbimg, Sphere, Scene, render, FancySphere
import GeometryTypes: Point, Vec3

include("img.jl")
include("nets/SqueezeNet/SqueezeNet.jl")
using .SqueezeNet_
include("prior.jl")
include("objects.jl")
# include("posterior.jl")

export img, scene, img_obs, img_real


# Optional
# using DataFrames
# using DataFrames
# using OmegaModels
# using Tensorboard
# using RunTools
# using Lens
# using ZenUtils
# include("diagnostics.jl")
# include("hyper.jl")
# include("samplediag.jl")

end