# Render at 224 by 224 because thats the size the neural network takes as input
rendersquare(x) = Img(RayTrace.render(x, width = 224, height = 224))

# Define Prior Distribution over Scenes (add 1 to ensure there is always at least one sphere)
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

"Show a scene"
showscene(scene) = rgb.(render(scene; width = 300, height = 300)')

# Sample from Prior
#nb showscene(rand(scene))

# Another Sample
#nb showscene(rand(scene))

"Example scene to create observed image"
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

# Show the observation
#nb showscene(obs_scene())

function sampleposterior(n = 1000)
  samples = rand(scene, img ==ₛ img_obs, n; alg = SSMH)
end

#nb scenesamples = sampleposterior()
#nb showscene(scenesamples[end])

function sampleposterior_noi(n = 50000; noi = false, alg = SSMH, gamma = 1.0, kwargs...)
  logdir = Random.randstring()
  noipred = Omega.lift(nointersect)(scene)
  obspred = img ==ₛ img_obs
  pred = noi ? (gamma * noipred) & obspred : obspred
  samples = rand(scene, pred, n, alg = alg, kwargs...)
  samples
end