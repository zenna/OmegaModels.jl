module BadCar

using AutoViz
using AutomotiveDrivingModels
using Omega
import Reel

const roadway = ciid(ω -> gen_straight_roadway(2, 100.0))

function scene_(ω)
  _roadway = roadway(ω)
  scene = Scene()

  p1 = uniform(ω, 0.0, 50.0)
  p2 = uniform(ω, 0.0, 50.0)
  
  push!(scene, Vehicle(VehicleState(VecSE2(p1, DEFAULT_LANE_WIDTH,0.0), _roadway, 29.0), VehicleDef(), 1))
  push!(scene, Vehicle(VehicleState(VecSE2(p2, 0.0 ,0.0), _roadway, 27.0), VehicleDef(), 2))
  scene
end

const scene = ciid(scene_)

# What is the inference problem?
function sim_(ω)
  _roadway = roadway(ω)
  # Set initial scene
  
  models = Dict{Int, LaneFollowingDriver}()
  # models[1] = StaticLaneFollowingDriver(0.0) # always produce zero acceleration
  models[1] = IntelligentDriverModel(v_des = 12.0) # default IDM with a desired speed of 12 m/s
  models[2] = PrincetonDriver(v_des = 10.0) # default Princeton driver with a desired speed of 10m/s

  nticks = 100
  timestep = 0.1
  rec = SceneRecord(nticks+1, timestep)
  simulate!(rec, scene(ω), _roadway, models, nticks)
  rec
end

const sim = ciid(sim_)

"Animate the model"
function animate_(ω; fps = 30, duration = 2)
  rec = sim(ω)
  cam = FitToContentCamera()
  car_colors = get_pastel_car_colors(scene(ω))
  _roadway = roadway(ω)

  # Linearly scale
  to_index(t) = Int(floor(t * 1/duration * nframes(rec))) + 1

  nframes_video = Int(floor(fps * duration))


  function r(t, dt)
    @show t
    # frame = t - nframes(rec)
    frame = to_index(t)
    render(rec.frames[frame],
          _roadway, 
          cam=cam,
          car_colors=car_colors)
  end
  # @manipulate for frame_index in 1 : nframes(rec)
  # end

  film = Reel.roll(r, fps = fps, duration = duration)
end

const animate = ciid(animate_)

function test()
  film = rand(animate)
  write("output.gif", film) 
end

end # module
