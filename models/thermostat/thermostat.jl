# Causal Modeling of time of day, ac, window, and thermostat
using Omega

# Create a model
# A uniform distribution over the time of day 
timeofday = uniform([:morning, :afternoon, :evening])

# The window is either open or not (with equal probability)
is_window_open = bernoulli(0.5, Bool)

# Turn off the a.c. when the window is closed!
const is_ac_on = ciid(rng -> is_window_open(rng[@id]) ? false : bernoulli(rng[@id], 0.5, Bool))

# The outside temperature is normally distributed with a mean depending on the time of day
function outside_temp_(rng)
  tod = timeofday(rng[@id])
  if tod == :morning
    normal(rng[@id], 20.0, 2.0)
  elseif tod == :afternoon
    normal(rng[@id], 32.0, 2.0)
  else
    normal(rng[@id], 10.0, 2.0)
  end
end

const outside_temp = ciid(outside_temp_)

# The inside temperature is normally distributed with a mean depending on whether the ac is on
function inside_temp_(rng)
  if is_ac_on(rng[@id])
    normal(rng[@id], 20.0, 2.0)
  else
    normal(rng[@id], 25.0, 2.0)
  end
end

const inside_temp = ciid(inside_temp_)

# The room is a perfect insulator: when the window is closed its only the inside temperature that matters
# Otherwise, its the mean of the outside and inside temperatures
function thermostat_(rng)
  if is_window_open(rng)
    (outside_temp(rng[@id]) + inside_temp(rng[@id])) / 2.0
  else
    inside_temp(rng[@id])
  end
end

const thermostat = ciid(thermostat_)

# Given the model, we can now define queries #
using UnicodePlots
using Plots
fontx = Plots.font("Helvetica", 20)
function plothist(samples; bins = 100, xlim = (0.0, 40.0))
  upscale = 8 #8x upscaling in resolution
  Plots.histogram(samples, bins = bins,
                  # bar_edges = true,
                  normalize=true,
                  # aspect_ratio = :equal,
                  size = (800, 600),
                  xlim = xlim,
                  # xticks = [0.0, 0.5, 1.0],
                  yticks = [],
                  xtickfont=fontx,
                  label="")
end

function plotbar(scenarios)
  counts(xs::Vector{T}) where T = (d = Dict{T, Int}(); foreach(x -> d[x] = get!(d, x, 0) + 1, xs); d)
  seccounts = counts(scenarios)
  Plots.bar(string.(collect((keys(seccounts)))), collect(values(seccounts)),
            normalize=true)
end

const allvars = (timeofday, is_window_open, is_ac_on, outside_temp, inside_temp, thermostat)

priorsamples = rand(outside_temp, 10000, alg = RejectionSample)
plothist(priorsamples)

# Conditional Inference: The thermostat reads hot. Shat does this tell you about outside temp?
priorsamplescond = rand(outside_temp, thermostat > 30, 10000, alg = RejectionSample)
plothist(priorsamplescond)

# You intervene on thermostat to be hot,  What does this tell you about outside temp (answer: nothing!)
outside_temp_do = replace(outside_temp, thermostat => 35.0)
priorsamplesdo = rand(outside_temp_do, 10000, alg = RejectionSample)
plothist(priorsamplesdo)

# Prior thermostat reading
thermopriorsamples = rand(thermostat, 100000, alg = RejectionSample)
plothist(thermopriorsamples, bins = 100, xlim = (10, 40))

# If I were to close the window and turn on the AC would that make it hotter or colder
thermostatnew = replace(thermostat, is_ac_on => true, is_window_open => false)
diffs = rand(thermostatnew - thermostat, 100000, alg = RejectionSample)
plothist(diffs, bins = 100)

# In what scenarios would it still be hotter after turning on the AC and closing the window?
scenarios = rand(timeofday, thermostatnew - thermostat > 0.0, 1000, alg = RejectionSample)
plotbar(scenarios)

# What if we opened the window and turned the AC on (logical inconsistency w.r.t to original model)
thermostat_imposs = replace(thermostat, is_ac_on => true, is_window_open => true)
samples_imposs = rand(thermostat_imposs, 100000, alg = RejectionSample)
plothist(samples_imposs, bins = 100, xlim = (10, 40))
# savefig("dothermoimposs.svg")

diffsamples_imposs = rand(thermostat_imposs - thermostat, 10000, alg = RejectionSample)
plothist(diffsamples_imposs, bins = 100, xlim = :auto)
mean(diffsamples_imposs)