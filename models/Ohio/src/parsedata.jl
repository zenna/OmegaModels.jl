module ParseData
using ExXML: Node, readxml
using Spec


struct Patient
  id::Int
  insulin_type
  weight
end

struct FingerStick{TS, VALUE}
  ts::TS
  value::VALUE
end

struct Basal{TS, VALUE}
  ts::TS
  value::VALUE
end

struct TempBasal{TSB, TSE, VALUE}
  ts_begin::TSB
  ts_end::TSE
  value::VALUE
end

struct Bolus{TSB, TSE, TYPE, DOSE, BWZ}
  ts_begin::TSB
  ts_end::TSE
  type::TYPE
  dose::DOSE
  bwz_carb_input::BWZ
end

struct Meal{TS, TYPE, CARBS}
  ts::TS
  type::TYPE
  carbs::CARBS
end

struct Sleep{TSB, TSE, QUALITY}
  ts_begin::TSB
  ts_end::TSE
  quality::QUALITY
end

struct Work{TSB, TSE, INTENSITY}
  ts_begin::TSB
  ts_end::TSE
  intensity::INTENSITY
end



function Patient(p::Node)
  @pre p.name == "patient"
  id = parse(Int, p["id"])
end



function event(x::Node, T = Float64)
  @pre x.name == "event"
  (ts =  DateTime(x["ts"], dateformat"d-m-y H:M:S"),
   value = parse(T, x["value"])) 
end

function glucoselevels(x::Node)
  @pre p1.name == "glucose_level"
  (glucose_level = map(event, elements(x),))
end

function smlly

nm(x) = Symbol(x.name)

function parseattr(attr::Node)
  key, val = attr.name, attr.content 
  if key == "id"
    parse(Int, val)
  elseif key in ["ts", "ts_begin", "ts_end", "tbegin", "tend"]
    DateTime(val, dateformat"d-m-y H:M:S")
  elseif key == "weight"
    parse(Int, val)
  elseif key == "insulin_type"
    val
  elseif key == "type"
    val
  elseif key == "dose"
    parse(Float64, val)
  elseif key == "bwz_carb_input"
    parse(Float64, val)
  elseif key == "carbs"
    parse(Float64, val)
  elseif key == "value"
    parse(Float64, val)
  elseif key == "quality"
    parse(Float64, val)
  elseif key == "intensity"
    parse(Float64, val)
  elseif key == "description"
    val
  elseif key == "duration"
    val
  elseif key == "competitive"
    val
  elseif key == "name"
    val
  else
    error("faileed to pasrse $key $val")
  end
end

function smelly(x::Node)
  nms = map(nm, attributes(x))
  attrs = map(parseattr, attributes(x))
  elements_ = elements(x)
  if isempty(elements_)
    NamedTuple{(nms...,)}((attrs...,))
  else
    vals = map(smelly, elements_)
    NamedTuple{(nms..., :vals)}((attrs..., vals))
  end
end

function test(;fn = "588-ws-training.xml")
  doc = readxml(fn)
end

end