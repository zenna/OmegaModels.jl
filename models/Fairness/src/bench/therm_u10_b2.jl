module Therm_u10_b2

using Omega
using ..Fairness

function pre()
    modal = step([(0,1,1/3),(1,2,1/3),(2,3,1/3)])
    if modal < 1
        lin = gaussian(30, 9)
    elseif modal < 2
        lin = gaussian(35, 9)
    else
        lin = gaussian(50, 9)
    end
    ltarget = gaussian(75, 1)
    return lin, ltarget
end    

function D(lin, ltarget)
    h = Hole(0, (0, 10))
    tOn = ltarget + Hole(0, (-10, 0))
    tOff = ltarget + Hole(0, (0, 10))
    isOn = 0
    K = 0.1
    curL = lin
    event("sanity1", tOn < tOff)
    event("sanity2", h > 0)
    event("sanity3", h < 20)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_0", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_1", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_2", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_3", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_4", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_5", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_6", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_7", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_8", curL < 120)
    if isOn > 0.5
        curL = curL + (h - K * (curL - lin))
        if curL > tOff
            isOn = 0
        end
    else
        curL = curL - K * (curL - lin)
        if curL < tOn
            isOn = 1
        end
    end
    event("body_9", curL < 120)
    Error = curL - ltarget
    if Error < 0
        Error = Error * -1
    end
    if Error < 2
        ret = 0
    else
        ret = 1
    end

    return ret

end

end