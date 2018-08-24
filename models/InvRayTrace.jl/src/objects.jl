using Omega
# using ImageView
using RunTools
using RayTrace
import RayTrace: SimpleSphere, ListScene, rgbimg
import RayTrace: FancySphere, Vec3, Sphere, Scene
using FileIO
using DataFrames

struct Img{T}
  img::T
end

# Render at 224 by 225 because that's what the neural networ expects
rendersquare(x) = Img(RayTrace.render(x, width = 224, height = 224))
rgbimg(x::Img) = rgbimg(x.img)

## Priors
## ======
const nspheres = poisson(3)

function sphere_(ω)
  FancySphere([uniform(ω[@id], -6.0, 6.0), uniform(ω[@id], -1.0, 1.0), uniform(ω[@id], -30.0, -10.0)],
               uniform(ω[@id], 1.0, 5.0),
              [uniform(ω[@id], 0.0, 1.0), uniform(ω[@id], 0.0, 1.0), uniform(ω[@id], 0.0, 1.0)],
               1.0,
               0.0,
               Vec3([0.0, 0.0, 0.0]))
end

"Randm Variable over scenes"
function scene_(ω)
  # spheres = map(1:nspheres(ω)) do i
  spheres = [sphere_(ω[i]) for i = 1:nspheres(ω)]
  base = FancySphere(Float64[0.0, -10004, -20], 10000.0, Float64[0.20, 0.20, 0.20], 0.0, 0.0, Float64[0.0, 0.0, 0.0])
  light = FancySphere(Vec3([0.0, 20.0, -30]),  3.0, Vec3([0.00, 0.00, 0.00]), 0.0, 0.0, Vec3([3.0, 3.0, 3.0]))
  push!(spheres, base)
  push!(spheres, light)  
  scene = ListScene(spheres)
end

"Show a random image"
showscene(scene) = imshow(rgbimg(render(scene)))

## Observation
## ===========
"Some example spheres which should create actual image"
function obs_scene()
  scene = [FancySphere(Float64[0.0, -10004, -20], 10000.0, Float64[0.20, 0.20, 0.20], 0.0, 0.0, Float64[0.0, 0.0, 0.0]),
           FancySphere(Float64[0.0,      0, -20],     4.0, Float64[1.00, 0.32, 0.36], 1.0, 0.0, Float64[0.0, 0.0, 0.0]),
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
  xfeatures = squeezenet2(expanddims(x.img))
  yfeatures = squeezenet2(expanddims(y.img))
  ds = map(eucl, xfeatures, yfeatures)
  lens(:scores, ds)
  mean(ds)
end
expanddims(x) = reshape(x, size(x)..., 1)

"Put image in channel width height form"
cwh(img) = permutedims(img, (3, 1, 2))

function sampleposterior()
  scene = ciid(scene_)                # Random Variable of scenes
  img = lift(rendersquare)(scene)     # Random Variable over images
  samples = rand(scene, img ==ₛ img_obs, 100; alg = SSMH, cb = cb)
end

function sampleposterioradv(n = 10000)
  scene = ciid(scene_)                # Random Variable of scenes
  img = lift(rendersquare)(scene)     # Random Variable over images

  logdir = Random.randstring()
  writer = Tensorboard.SummaryWriter(logdir)
  cb = cbs(writer, logdir, n, img)
  lmap = lenses(writer)
  lenscall(lmap, rand, scene, img ==ₛ img_obs, n; alg = SSMH, cb = cb)
end