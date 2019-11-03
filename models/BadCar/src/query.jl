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

# Sample from the distribution over modes conditioned o
# Observed car movemennt
function query1(; n = 10_000, nchanges = 10.0)
  # @assert false
  rand(Ω, cond(dynamic_sim, n_lane_change ==ₛ nchanges), n; alg = Replica)
end

politeness(ω, driver_id = 1) = models(ω)[driver_id].mlane.politeness

pol = ~politeness

# function make_plots()
#   # samplemean(n_lane_change, 1)  
#   # # expected_lane_changes = 5.0
#   n = 10_000

#   samples1 = @leval SSMHLoop => Omega.default_cbs(n) BadCar.query1(; n = n, nchanges = 10.0)
#   plt1 = Plots.histogram(BadCar.politeness.(samples1[500:end]), xlabel = "politeness | nchanges = 10", nbins = 10, xlim = [0, 1], legend = false, title = "Posterior over politeness given 10 changes")
#   savefig(plt1, "plot1.pdf")

#   samples2 = @leval SSMHLoop => Omega.default_cbs(n) BadCar.query1(; n = n, nchanges = 3.0)
#   plt2 = Plots.histogram(BadCar.politeness.(samples2[500:end]), xlabel = "politeness | nchanges = 3", nbins = 10, xlim = [0, 1], legend = false, title = "Posterior over politeness given 3 changes")
#   savefig(plt2, "plot2.pdf")

#   samples1, samples2

#   # samples3 = rand(BadCar.pol, n)
#   # Plots.histogram(samples3[5:end], xlabel = "politeness", nbins = 10, xlim = [0, 1], legend = false, title = "Prior over politeness")
# end 