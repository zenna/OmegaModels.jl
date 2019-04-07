using Omega
# using ImageView
using RunTools
using RayTrace
import RayTrace
import RayTrace: ListScene, rgbimg, rgb, msphere, Vec3, Sphere, Scene, render, MaterialGeom
import GeometryTypes: Point, Vec3
using FileIO
using DataFrames

"Put image in channel width height form"
cwh(img) = (@show typeof(img); permutedims(img, (3, 1, 2)))
function cube(ii)
  a = Array{Float64}(undef, (size(ii)...),3)
  for i = 1:size(a, 1), j = 1:size(a, 2)
    a[i,j,:] = ii[i,j]
  end
  a
end

struct Img{T}
  img::T
end

# Render at 224 by 225 because
rendersquare(x) = Img(RayTrace.render(x, width = 224, height = 224))
rgbimg(x::Img) = rgbimg(x.img)

# Define Prior Distribution over Scenes
const nspheres = poisson(3) + 1

"Random Variable over Spheres"
function sphere_(ω)
  msphere(Point(uniform(ω[@id], -6.0, 6.0), uniform(ω[@id], -1.0, 1.0), uniform(ω[@id], -30.0, -10.0)),
          uniform(ω[@id], 1.0, 5.0),
          Vec3(uniform(ω[@id], 0.0, 1.0), uniform(ω[@id], 0.0, 1.0), uniform(ω[@id], 0.0, 1.0)),
          1.0,
          0.0,
          Vec3(0.0, 0.0, 0.0))
end

"Random Variable over scenes"
function scene_(ω)
  spheres = [sphere_(ω[i]) for i = 1:nspheres(ω)]
  base = msphere(Point(0.0, -10004, -20), 10000.0, Vec3(0.20, 0.20, 0.20), 0.0, 0.0, Vec3(0.0, 0.0, 0.0))
  light = msphere(Point(0.0, 20.0, -30), 3.0, zeros(Vec3), 0.0, 0.0, Vec3(3.0, 3.0, 3.0))
  push!(spheres, base)
  push!(spheres, light)  
  scene = ListScene(spheres)
end
const scene = ciid(scene_)                # Random Variable of scenes
const img = lift(rendersquare)(scene)     # Prior distribution over images

"Show a random image"
showscene(scene) = imshow(rgbimg(render(scene)))

# Observation #
"Some example spheres which should create actual image"
function obs_scene()
  scene = [msphere(Point(0.0, -10004, -20), 10000.0, Vec3(0.20, 0.20, 0.20), 0.0, 0.0, Vec3(0.0, 0.0, 0.0)),
           msphere(Point(0.0, 0.0, -20), 4.0, Vec3(1.0, 0.32, 0.36), 1.0, 0.5, zeros(Vec3)),
           msphere(Point(5.0, 1.0, -15), 2.0, Vec3(0.90, 0.76, 0.46), 1.0, 0.0, zeros(Vec3)),
           msphere(Point(5.0, 0.0, -25), 3.0, Vec3(0.65, 0.77, 0.970), 1.0, 0.0, zeros(Vec3)),
           msphere(Point(-5.5,      0, -15), 3.0, Vec3(0.90, 0.90, 0.90), 1.0, 0.0, zeros(Vec3)),
           msphere(Point(0.0, 20.0, -30), 3.0, zeros(Vec3), 0.0, 0.0, Vec3(3.0, 3.0, 3.0))]
  RayTrace.ListScene(scene)
end

const img_obs = rendersquare(obs_scene())

# Equality #
eucl(x, y) = sqrt(sum((x - y) .^ 2))
function Omega.d(x::Img, y::Img)::Real
  xfeatures = squeezenet2(expanddims(cube(x.img)))
  yfeatures = squeezenet2(expanddims(cube(y.img)))
  eucl(xfeatures[1], yfeatures[1])
  # ds = map(eucl, xfeatures, yfeatures)
  # lens(:scores, ds)
  # mean(ds)
end
expanddims(x) = reshape(x, size(x)..., 1)

function sampleposterior(n = 1000)
  # img = lift(rendersquare)(scene)     # Random Variable over images
  samples = rand(scene, img ==ₛ img_obs, n; alg = SSMH)
end

function sampleposterioradv(n = 50000; noi = false, alg = SSMH, gamma = 1.0, kwargs...)
  logdir = Random.randstring()
  writer = Tensorboard.SummaryWriter(logdir)
  cb = cbs(writer, logdir, n, img)
  noipred = Omega.lift(nointersect)(scene)
  obspred = img ==ₛ img_obs
  pred = noi ? (gamma * noipred) & obspred : obspred
  samples = rand(scene, pred, n; cb = cb, alg = alg, kwargs...)
  lmap = lenses(writer)
  samples
  # lenscall(lmap, rand, scene, img ==ₛ img_obs, n; alg = SSMH, cb = cb)
end