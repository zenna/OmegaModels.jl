module DT16
using ..Fairness

function pre(ω)
    sex = steps(ω, [(0,1,0.3307), (1,2,0.6693)])
    if sex < 1
        capital_gain = gaussian(ω, 568.4105, 24248365.5428)
        if capital_gain < 7298.0000
            relationship = steps(ω, [(0,1,0.0491), (1,2,0.1556), (2,3,0.4012), (3,4,0.2589), (4,5,0.0294), (5,6,0.1058)])
        else
            relationship = steps(ω, [(0,1,0.0416), (1,2,0.1667), (2,3,0.4583), (3,4,0.2292), (4,5,0.0166), (5,6,0.0876)])
        end
    else
        capital_gain = gaussian(ω, 1329.3700, 69327473.1006)
        if capital_gain < 5178.0000
            relationship = steps(ω, [(0,1,0.0497), (1,2,0.1545), (2,3,0.4021), (3,4,0.2590), (4,5,0.0294), (5,6,0.1053)])
        else
            relationship = steps(ω, [(0,1,0.0417), (1,2,0.1624), (2,3,0.3976), (3,4,0.2606), (4,5,0.0356), (5,6,0.1021)])
        end
    end
    return sex, capital_gain, relationship
end

const prerv = ~pre
const Holes1 = (~ ω -> Hole(ω, 5095.5; σ = 0.0001),
               ~ ω -> Hole(ω, 4718.5; σ = 0.0001),
               ~ ω -> Hole(ω, 5095.5; σ = 0.0001),
               ~ ω -> Hole(ω, 8296.0; σ = 0.0001),
               ~ ω -> Hole(ω, 4668.5; σ = 0.0001))

const Holes2 = (~ ω -> Hole(ω, 0.0; σ = 0.001),
               ~ ω -> Hole(ω, 2632.1803124654944; σ = 0.001),
               ~ ω -> Hole(ω, 1168.5702749404127; σ = 0.001),
               ~ ω -> Hole(ω, 20229.055378016285; σ = 0.001),
               ~ ω -> Hole(ω, 4108.646587752379; σ = 0.001))

# const Holes = (~ ω -> Hole(ω, rand(); σ = 0.001),
#                 ~ ω -> Hole(ω, rand(); σ = 0.001),
#                 ~ ω -> Hole(ω, rand(); σ = 0.001),
#                 ~ ω -> Hole(ω, rand(); σ = 0.001),
#                 ~ ω -> Hole(ω, rand(); σ = 0.001))

               
function D(Holes, sex, capital_gain, relationship)
    event("minority", sex < 1)
    # @show relationship
    if relationship < 1
        # if capital_gain < Hole(ω, 5095.5; σ = 200.0)
        # if capital_gain < Hole(ω, 0.0; σ = 0.0)
        if capital_gain < Holes[1]
            t = 1
        else
            t = 0
        end
    elseif relationship < 2
        # if capital_gain < Hole(ω, 4718.5; σ = 200.0)
        # if capital_gain < Hole(ω, 2632.1803124654944; σ = 0.0)
        if capital_gain < Holes[2]
            # @show "failed"
            t = 1
        else
            # @show "won"
            t = 0
        end
    elseif relationship < 3
        # if capital_gain < Hole(ω, 5095.5; σ = 200.0)
        # if capital_gain < Hole(ω, 1168.5702749404127; σ = 0.0)
        if capital_gain < Holes[3]
            t = 1
        else
            t = 0
        end
    elseif relationship < 4
        # if capital_gain < Hole(ω, 8296); σ = 200.0
        # if capital_gain < Hole(ω, 20229.055378016285; σ = 0.0)
        if capital_gain < Holes[4]
            t = 1
        else
            t = 0
        end
    elseif relationship < 5
        t = 1
    else
        # if capital_gain < Hole(ω, 4668.5; σ = 200.0)
        # if capital_gain < Hole(ω, 4108.646587752379; σ = 0.0)
        if capital_gain < Holes[5]
            t = 1
        else
            t = 0
        end
    end
    event("hired", t < 0.5)
    ret = 1 - t
    t
    # return ret
end

function classifier_(ω; Holes = Holes1)
    Holes_ = map(h->h(ω), Holes)
    let w_ = ω
        function (data)
            D(Holes_, data.sex, data[Symbol("capital-gain")], data.relationship)
        end
    end
end

const classifier = ~ classifier_

# function post(Pr, hired, minority)
#     num = prob(hired & minority) / prob(minority)
#     den = prob(hired & !minority) / prob(!minority)
#     ratio = num / den
#     ratio > 0.85
# end

function post(hired, minority, theta)
    num = prob(rid(hired & minority, theta)) / prob(minority)
    den = prob(rid(hired & !minority, theta)) / prob(!minority)
    ratio = num / den
    #                ~ ω -> Hole(ω, 2632.1803124654944; σ = 0.001),
    #                ~ ω -> Hole(ω, 1168.5702749404127; σ = 0.001),
    #                ~ ω -> Hole(ω, 20229.055378016285; σ = 0.001),
    #                ~ ω -> Hole(ω, 4108.646587752379;
    ratio > 0.85
end

# post(hired, minority, Holes)
# 
# @show Fairnes
# @show Fairness.evaluate(Fairness.load_data(), Fairness.exp_decision(Fairness.DT16.classifier; nsamples = 1))

end