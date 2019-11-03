
function scene_(ω)
  _roadway = roadway(ω)
  scene = Scene()

  p1 = uniform(ω, 0.0, 50.0)
  p2 = uniform(ω, 0.0, 50.0)
  
  push!(scene, Vehicle(VehicleState(VecSE2(p1, DEFAULT_LANE_WIDTH,0.0), _roadway, 29.0), VehicleDef(), 1))
  push!(scene, Vehicle(VehicleState(VecSE2(p2, 0.0 ,0.0), _roadway, 27.0), VehicleDef(), 2))
  scene
end

scene = ~ scene_

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

sim =~ sim_
