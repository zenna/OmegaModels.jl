using StatsBase

function evalposterior(samples, realvideo, verbose=false, visual=false)
  distancematrix = [surjection(realvideo[j].objects, samples[x][j].objects) for j=1:length(realvideo), x=1:length(samples)]
  # Part 1: Get average distance of each sample.
  sampleperformance = [mean(distancematrix[:,i]) for i=1:size(distancematrix)[2]]
  avgperformance = round(mean(sampleperformance), digits = 2)
  if verbose println("Average sample distance = $avgperformance\n") end
  if visual println(histogram(sampleperformance, title="Average distance of each sample")) end
  # Part 2: Find best and worst sample.
  bestsample = findmin(sampleperformance)
  bestsampleno = bestsample[2]
  bestsampled = round(bestsample[1], digits = 2)
  if verbose println("Best sample's (#$bestsampleno) avg. dist = $bestsampled\n") end
  worstsample = findmax(sampleperformance)
  worstsampleno = worstsample[2]
  worstsampled = round(worstsample[1], digits = 2)
  if verbose println("Worst sample's (#$worstsampleno) avg. dist = $worstsampled\n") end
  # Part 3: Get average distances over time
  timeperformance = [mean(distancematrix[i,:]) for i=1:size(distancematrix)[1]]
  timecor = round(corspearman(1:length(realvideo),timeperformance), digits = 2)
  if verbose println("Spearman with time = $timecor\n") end
  if visual println(lineplot(1:length(realvideo),timeperformance, title="Average distance of samples over time")) end
  results = Dict(:avgdistance => avgperformance, :bestdistance => bestsampled, :bestdistanceid => bestsampleno, :worstdistance => worstsampled, :worstdistanceid => worstsampleno, :timecor => timecor)
  return results
end