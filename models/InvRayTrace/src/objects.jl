# Render at 224 by 224 because thats the size the neural network takes as input
rendersquare(x) = Img(RayTrace.render(x, width = 224, height = 224))

# Define Prior Distribution over Scenes (add 1 to ensure there is always at least one sphere)
# const nspheres = (@~ Poisson(3)) .+ 1
const nspheres = Variable(ω -> 4)

"Random Variable over Spheres"
function sphere_(i, ω::Ω)
  FancySphere(Point(((@uid, i)~ Uniform(-6.0, 6.0))(ω), ((@uid, i)~ Uniform(-1.0, 1.0))(ω), ((@uid, i)~ Uniform(-30.0, -10.0))(ω)),
  ((@uid, i)~ Uniform(1.0, 5.0))(ω),
          Vec3(((@uid, i)~ Uniform(0.0, 1.0))(ω), ((@uid, i)~ Uniform(0.0, 1.0))(ω), ((@uid, i)~ Uniform(0.0, 1.0))(ω)),
          1.0,
          0.0,
          Vec3(0.0, 0.0, 0.0))
end

"Random Variable over scenes"
function scene(ω::Ω)
  spheres = [sphere_(i, ω) for i = 1:nspheres(ω)]
  base = FancySphere(Point(0.0, -10004, -20), 10000.0, Vec3(0.20, 0.20, 0.20), 0.0, 0.0, Vec3(0.0, 0.0, 0.0))
  light = FancySphere(Point(0.0, 20.0, -30), 3.0, zeros(Vec3), 0.0, 0.0, Vec3(3.0, 3.0, 3.0))
  push!(spheres, base)
  push!(spheres, light)  
  scene = ListScene(spheres)
end
const img = pw(rendersquare, scene)     # Prior distribution over images

"Example scene to create observed image"
function obs_scene()
  scene = [FancySphere(Point(0.0, -10004, -20), 10000.0, Vec3(0.20, 0.20, 0.20), 0.0, 0.0, Vec3(0.0, 0.0, 0.0)),
           FancySphere(Point(0.0, 0.0, -20), 4.0, Vec3(1.0, 0.32, 0.36), 1.0, 0.5, zeros(Vec3)),
           FancySphere(Point(5.0, 1.0, -15), 2.0, Vec3(0.90, 0.76, 0.46), 1.0, 0.0, zeros(Vec3)),
           FancySphere(Point(5.0, 0.0, -25), 3.0, Vec3(0.65, 0.77, 0.970), 1.0, 0.0, zeros(Vec3)),
           FancySphere(Point(-5.5,      0, -15), 3.0, Vec3(0.90, 0.90, 0.90), 1.0, 0.0, zeros(Vec3)),
           FancySphere(Point(0.0, 20.0, -30), 3.0, zeros(Vec3), 0.0, 0.0, Vec3(3.0, 3.0, 3.0))]
  RayTrace.ListScene(scene)
end

const img_obs = rendersquare(obs_scene())

# load the real image
i = Images.load("boxes.jpg")
ii = imresize(i, (224, 224))
b1 = convert(Array{Float32}, channelview(ii))
b2 = Array{Float32, 3}(undef, 224, 224, 3)
for i in 1:3
    b2[:, :, i] = b1[i, :, :]
end
const img_real = InvRayTrace.Img(b2)