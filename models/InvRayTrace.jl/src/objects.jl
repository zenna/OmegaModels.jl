using Omega
# using ImageView
using RunTools
using RayTrace
import RayTrace: SimpleSphere, ListScene, rgbimg
import RayTrace: FancySphere, Vec3, Sphere, Scene
using FileIO
using DataFrames

include("net.jl")

struct Img{T}
  img::T
end

# Render at 224 by 225 because that's what the neural networ expects
rendersquare(x) = Img(RayTrace.render(x, width = 224, height = 224))
rgbimg(x::Img) = rgbimg(x.img)

## Priors
## ======
const nspheres = poisson(3)
"Randm Variable over scenes"
function scene_(ω)
  # spheres = map(1:nspheres(ω)) do i
  spheres = map(1:4) do i
    FancySphere([uniform(ω[@id][i], -6.0, 5.0), uniform(ω[@id][i] , -1.0, 0.0), uniform(ω[@id][i]  , -25.0, -15.0)],
                 uniform(ω[@id][i]  , 1.0, 4.0),
                 [uniform(ω[@id][i] , 0.0, 1.0), uniform(ω[@id][i] , 0.0, 1.0), uniform(ω[@id][i] , 0.0, 1.0)],
                 1.0,
                 0.0,
                 Vec3([0.0, 0.0, 0.0]))
  end
  light = FancySphere(Vec3([0.0, 20.0, -30]),  3.0, Vec3([0.00, 0.00, 0.00]), 0.0, 0.0, Vec3([3.0, 3.0, 3.0]))
  push!(spheres, light)
  scene = ListScene(spheres)
end

# "Randm Variable over scenes"
# function scene_(ω)
#   # spheres = map(1:nspheres(ω)) do i
#   spheres = map(1:4) do i
#     FancySphere([uniform(ω[@id][i], -6.0, 5.0), uniform(ω[@id][i] , -6.0, 0.0), uniform(ω[@id][i]  , -25.0, -15.0)],
#                 #  uniform(ω[@id][i]  , 1.0, 4.0),
#                  1.0,
#                  [uniform(ω[@id][i] , 0.0, 1.0), uniform(ω[@id][i] , 0.0, 1.0), uniform(ω[@id][i] , 0.0, 1.0)],
#                  1.0,
#                  0.0,
#                  Vec3([0.0, 0.0, 0.0]))
#   end
#   light = FancySphere(Vec3([0.0, 20.0, -30]),  3.0, Vec3([0.00, 0.00, 0.00]), 0.0, 0.0, Vec3([3.0, 3.0, 3.0]))
#   # push!(spheres, light) 4
#   scene = ListScene([spheres; light])
# end

# "Randm Variable over scenes"
# function scene_(ω)
#   # spheres = map(1:nspheres(ω)) do i
#   spheres = map(1:10) do i
    
#     FancySphere(uniform(ω[@id][i], 0.0, 1.0, (3,)),
#                  0.5,
#                  uniform(ω[@id][i], 0.0, 1.0, (3,)),
#                  1.0,
#                  0.0,
#                  Vec3([0.0, 0.0, 0.0]))
#   end
#   light = FancySphere(Vec3([0.0, 20.0, -30]),  3.0, Vec3([0.00, 0.00, 0.00]), 0.0, 0.0, Vec3([3.0, 3.0, 3.0]))
#   # push!(spheres, light)  
#   scene = ListScene([spheres; light])
# end

"Show a random image"
showscene(scene) = imshow(rgbimg(render(scene)))

## Observation
## ===========
"Some example spheres which should create actual image"
function obs_scene()
  scene = [FancySphere(Float64[0.0, -10004, -20], 10000.0, Float64[0.20, 0.20, 0.20], 0.0, 0.0, Float64[0.0, 0.0, 0.0]),
           FancySphere(Float64[0.0,      0, -20],     4.0, Float64[1.00, 0.32, 0.36], 1.0, 0.5, Float64[0.0, 0.0, 0.0]),
           FancySphere(Float64[5.0,     -1, -15],     2.0, Float64[0.90, 0.76, 0.46], 1.0, 0.0, Float64[0.0, 0.0, 0.0]),
           FancySphere(Float64[5.0,      0, -25],     3.0, Float64[0.65, 0.77, 0.97], 1.0, 0.0, Float64[0.0, 0.0, 0.0]),
           FancySphere(Float64[-5.5,      0, -15],    3.0, Float64[0.90, 0.90, 0.90], 1.0, 0.0, Float64[0.0, 0.0, 0.0]),
           # light (emission > 0)
           FancySphere(Float64[0.0,     20.0, -30],  3.0, Float64[0.00, 0.00, 0.00], 0.0, 0.0, Float64[3.0, 3.0, 3.0])]
  RayTrace.ListScene(scene)
end

const img_obs = rendersquare(obs_scene())

## Equality
## ========
eucl(x, y) = sqrt(sum((x - y) .^ 2))
function Omega.d(x::Img, y::Img)
  xfeatures = squeezenet(expanddims(x.img))
  yfeatures = squeezenet(expanddims(y.img))
  ds = map(eucl, xfeatures, yfeatures)
  mean(ds)
end
expanddims(x) = reshape(x, size(x)..., 1)

function sampleposterior()
  scene = ciid(scene_)                # Random Variable of scenes
  img = lift(rendersquare)(scene)     # Random Variable over images
  samples = rand(scene, img ==ₛ img_obs, 100; alg = SSMH, cb = cb)
end

function sampleposterioradv()
  scene = ciid(scene_)                # Random Variable of scenes
  img = lift(rendersquare)(scene)     # Random Variable over images
  cbh = (data, stage) -> addhausdorff(data, stage; groundtruth = obs_scene())
  cb = idcb → cbh →  plotscalar(:hausdorff, "Hausdorff distance between scenes")

  # View images
  writer = Tensorboard.SummaryWriter("./")
  tb_imgs(imgs...) = foreach(((i, img),) -> add_image!(writer, "l$i", img[:, :, 1]), enumerate(imgs))
  lmap = (filters = tb_imgs,)
  lenscall(lmap, rand, scene, img ==ₛ img_obs, 100; alg = SSMH, cb = cb)
end