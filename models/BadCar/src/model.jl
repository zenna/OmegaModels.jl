# roadway = ~ ω -> gen_straight_roadway(2, 1000.0)
roadway = ~ ω -> gen_stadium_roadway(2; length = 50.0)

ncars = poisson(4.0) + 1

function dynamic_scene_(ω)
  _roadway = roadway(ω)
  _scene = Scene()
  _ncars = ncars(ω)
  for i = 1:_ncars
    p = uniform(ω, 0.0, 100.0)
    l = uniform(ω, 0.0, DEFAULT_LANE_WIDTH)
    # l = uniform(ω, [DEFAULT_LANE_WIDTH, 0.0])
    push!(_scene, Vehicle(VehicleState(VecSE2(p, l, 0.0), _roadway, 29.0), VehicleDef(), i))
  end
  _scene
end

dynamic_scene = ~ dynamic_scene_

function models_(ω, timestep = 0.1)
  # models = Dict{Int, LaneFollowingDriver}()
  models = Dict{Int, Tim2DDriver}()
  for i = 1:ncars(ω)
    v_des = uniform(ω, 5.0, 15.0)
    # models[i] = IntelligentDriverModel(v_des = 12.0)
    models[i] = Tim2DDriver(timestep; mlon = IntelligentDriverModel(v_des = v_des))
    # # models[1] = StaticLaneFollowingDriver(0.0) # always produce zero acceleration
    # models[1] = IntelligentDriverModel(v_des = 12.0) # default IDM with a desired speed of 12 m/s
    # models[2] = PrincetonDriver(v_des = 10.0) # default Princeton driver with a desired speed of 10m/s
  end
  models
end

models = ~ models_

# What is the inference problem?
function dynamic_sim_(ω; nticks = 500, timestep = 0.1)
  _roadway = roadway(ω)

  # Set initial scene
  rec = SceneRecord(nticks+1, timestep)
  simulate!(rec, dynamic_scene(ω), _roadway, models(ω), nticks)
  rec
end

dynamic_sim = ~ dynamic_sim_


"Animate the model"
function animate_(ω; sim = dynamic_sim, fps = 60, duration = 5)
  rec = dynamic_sim(ω)
  cam = FitToContentCamera()
  car_colors = get_pastel_car_colors(dynamic_scene(ω))
  _roadway = roadway(ω)

  # Linearly scale
  to_index(t) = Int(floor(t * 1/duration * nframes(rec))) + 1

  nframes_video = Int(floor(fps * duration))
  
  function r(t, dt)
    @show t
    # frame = t - nframes(rec)
    @show frame_index = to_index(t)
    scene = rec[frame_index-nframes(rec)]

    # Add an overlay of id above the car
    overlays = [TextOverlay(text = ["$(veh.id)"],
            incameraframe = true,
            pos = veh.state.posG  + VecSE2(0.0, 1.0, 1.0)) for veh in scene]

    render(scene,
           _roadway, 
           cam = cam,
           car_colors = car_colors)
  end

  film = Reel.roll(r, fps = fps, duration = duration)
end

animate = ciid(animate_)