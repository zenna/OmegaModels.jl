function plot_prior_samples(ωs; nsamples = 10, kwargs...)
  ωs_sub = rand(ωs, nsamples)
  sims = [co2_sim(ω) for ω in ωs_sub]
  plot(1959:2035, sims;
       title = "Prior Samples",
       xlabel = "Year",
       ylabel = "CO2 (ppm)",
       legend = false,
       color = :red,
       kwargs...)
end

function plot_posterior_samples(ωs; nsamples = 10, kwargs...)
  ωs_sub = rand(ωs, nsamples)
  sims = [co2_sim(ω) for ω in ωs_sub]
  plt = plot(1959:2035, sims;
       title = "Prior Samples",
       xlabel = "Year",
       ylabel = "CO2 (ppm)",
       legend = false,
       color = :red,
       kwargs...)
  # Add co2 data
  cd = asvec(co2_data, 1959:2018)
  scatter!(plt, 1959:2018, cd)
  plt
end

function dataplot()
  td = asvec(temp_data, 1959:2018)
  cd = asvec(co2_data, 1959:2018)
  plt_t = plot(1959:2018, td, xlabel = "time", labels = "Global Temperature Anomaly (C)", framestyle = :box, tickfontsize = 16)
  plt_c = plot(1959:2018, cd, xlabel = "time", labels = "CO2 (ppm)", mirror = true, color = :red, framestyle = :box, tickfontsize = 16)
  plt_c, plt_t
  # fts = allfontsizes(SIGPLAN, PGFPlotsBackend; dpi = 300)
  # plt = Plots.plot(plts...;  size = (800, 500), dpi = 300, fts...)
end

