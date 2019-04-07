__precompile__(false)
"Inverse Graphics"
module InvRayTrace

using Omega
using RayTrace
using JLD2
using FileIO
using Flux
using DataFrames
import Statistics: mean
using Random
using Callbacks

import RayTrace: ListScene, rgbimg, rgb, msphere, Vec3, Sphere, Scene, render, MaterialGeom
# import GeometryTypes: Point, Vec3
using FileIO

include("img.jl")
include("net.jl")
include("prior.jl")

# export img, scene, img_obs

# include("notebook.jl")

# Optional
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