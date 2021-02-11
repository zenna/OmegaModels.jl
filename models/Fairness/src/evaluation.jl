# Evaluate a model
"""Evaluate the `model` on the dataset

Arguments
`dataset` - something with iterable elements, each corresponding to one datapoint
`model` - function which maps dataelement to decision

"""
function evaluate(dataset, classifier)
  ncorrect = 0
  total = 0
  isrich(x::Int) = x - 1
  for row in skipmissing(eachrow(dataset))
    input = row[1:14]
    # @show isrich(row[:label])
    label = isrich(row[:label])
    if ismissing(label)
      continue
    end
    # @show label
    # @show classifier(input)
    if classifier(input) == label
      ncorrect += 1
    end
    total += 1
  end
  ncorrect / total
end

function exp_decision(model; nsamples = 10)
  function(datum)
    samples = [rand(model)(datum) for i = 1:nsamples]
    # @show samples
    round(mean(samples))  # True if average is greater than 0.5
  end
end

# Test
const rand_model = ~ ω -> let b = bernoulli(ω, 0.5, Bool); (args...) -> b; end
