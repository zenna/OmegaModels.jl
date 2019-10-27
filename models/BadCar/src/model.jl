# roadway = ~ ω -> gen_straight_roadway(2, 1000.0)
roadway = ~ ω -> gen_stadium_roadway(2; length = 30.0)

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

# dynamic_scene = ~ dynamic_scene_

vehicledef_(ω) = VehicleDef(; class = 1)

"Generate scene with cars lined up like a race"
function gen_scene(ω, id = 1; gap = 3.0)
  _roadway = roadway(ω)
  _ncars = ncars(ω)
  rs = _roadway.segments[id]

  curr_lane = 0
  _nlanes = length(rs.lanes)
  cx = Dict(i => 0.0 for i = 1:_nlanes)

  _scene = Scene()
  for i = 1:_ncars
    vd = vehicledef_(ω)
    v = 0.0
    posF = Frenet(rs.lanes[curr_lane + 1], cx[curr_lane + 1])
    vs = VehicleState(posF, _roadway, v)
    veh = Vehicle(vs, vd, i)

    cx[curr_lane + 1] += vd.length + gap
    curr_lane = (curr_lane + 1) % _nlanes

    push!(_scene, veh)
  end
  _scene
end

dynamic_scene =~ gen_scene

function models_(ω, timestep = 0.1)
  # models = Dict{Int, LaneFollowingDriver}()
  models = Dict{Int, Tim2DDriver}()
  for i = 1:ncars(ω)
    v_des = uniform(ω, 1.0, 15.0)
    # models[i] = IntelligentDriverModel(v_des = 12.0)
    # models[i] = Tim2DDriver(timestep)
    models[i] = Tim2DDriver(timestep; mlon = IntelligentDriverModel(v_des = v_des),
                                      mlane = MOBIL(0.2; politeness = 0.1))
    # # models[1] = StaticLaneFollowingDriver(0.0) # always produce zero acceleration
    # models[1] = IntelligentDriverModel(v_des = 12.0) # default IDM with a desired speed of 12 m/s
    # models[2] = PrincetonDriver(v_des = 10.0) # default Princeton driver with a desired speed of 10m/s
  end
  models
end

models = ~ models_

# What is the inference problem?
function dynamic_sim_(ω; nticks = 1000, timestep = 0.1)
  _roadway = roadway(ω)

  # Set initial scene
  rec = SceneRecord(nticks+1, timestep)
  simulate!(rec, dynamic_scene(ω), _roadway, models(ω), nticks)
  rec
end

dynamic_sim = ~ dynamic_sim_

"Animate `rec` the model"
function animate(rec, roadway; fps = 60, duration = 5)
  cam = FitToContentCamera()

  scene = rec.frames[1]
  car_colors = get_pastel_car_colors(scene)

  # Linearly scale
  to_index(t) = Int(floor(t * 1/duration * nframes(rec))) + 1

  nframes_video = Int(floor(fps * duration))

  
  function r(t, dt)
    frame_index = to_index(t)
    scene = rec[frame_index-nframes(rec)]

    # Add an overlay of id above the car
    overlays = [TextOverlay(text = ["$(veh.id)"],
            incameraframe = true,
            pos = veh.state.posG  + VecSE2(0.0, 1.0, 1.0)) for veh in scene]

    render(scene,
           roadway, 
           cam = cam,
           car_colors = car_colors)
  end

  film = Reel.roll(r, fps = fps, duration = duration)
end

animated = lift(animate)(dynamic_sim, roadway)

# animate = ciid(animate_)