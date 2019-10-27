"Lane Tags of ith car"
function lane_tags_(ω, i)
  ds = dynamic_sim(ω)
  r = roadway(ω)
  [get_lane(r, ds.frames[j][i]).tag for j = 1:nframes(ds)]
end

driver_id = 1

lane_tag = ciid(lane_tags_, driver_id)

"Number of lane changes in sequence of lanetags"
function n_lane_changes(lanetags)
  lanes = map(lanetag -> lanetag.lane, lanetags)
  nchanges = 0
  curr_lane = first(lanes)
  for lane in lanes
    if lane != curr_lane
      nchanges += 1
      curr_lane = lane
    end
  end
  nchanges
end

n_lane_change = lift(n_lane_changes)(lane_tag)

"Lane Tags of ith car"
function (ω, i)
  ds = dynamic_sim(ω)
  r = roadway(ω)
  [get_lane(r, ds.frames[j][i]).tag for j = 1:nframes(ds)]
end

lane_tag = ciid(lane_tags_, 1)

# Sample from the distribution over modes conditioned o
# Observed car movement
function query1()
  function condition(ω)
    @show a = n_lane_change(ω)
    @show a ==ₛ 10.0
  end
  rand((dynamic_sim), ~condition, 1000; alg = SSMH)
end

function query2()
  rand()
end