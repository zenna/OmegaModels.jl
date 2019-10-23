# Create a stretch with two lanes
nlanes = 3
roadway = gen_straight_roadway(nlanes, 100.0)

# Put some cars on the road
# car = ArrowCar([50.0, 0.0], 0.0, color=colorant"blue") # [north, east], angle

# Set initial scene
scene = Scene()
push!(scene, Vehicle(VehicleState(VecSE2(20.0, DEFAULT_LANE_WIDTH,0.0), roadway, 29.0), VehicleDef(), 1))
push!(scene, Vehicle(VehicleState(VecSE2(70.0, 0.0 ,0.0), roadway, 27.0), VehicleDef(), 2))

car_colors = get_pastel_car_colors(scene)
cam = FitToContentCamera()
render(scene, roadway, cam=cam, car_colors=car_colors)

# Add Car Models
models = Dict{Int, LaneFollowingDriver}()
# models[1] = StaticLaneFollowingDriver(0.0) # always produce zero acceleration
models[1] = IntelligentDriverModel(v_des=12.0) # default IDM with a desired speed of 12 m/s
models[2] = PrincetonDriver(v_des = 10.0) # default Princeton driver with a desired speed of 10m/s


# Simulate the scene

# Rendering
nticks = 100
timestep = 0.1
rec = SceneRecord(nticks+1, timestep)
simulate!(rec, scene, roadway, models, nticks)
render(rec[0], roadway, cam=cam, car_colors=car_colors)

# 

# The objective of this model is to infer the model from the behavior of a car
# You are driving on the road, when you observe the behaviour of a car behind you
# You want to infer what kind of driver s/he is

# Let's setup a driving simulation



