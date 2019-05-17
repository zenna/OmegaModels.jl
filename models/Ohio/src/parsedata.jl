module ParseData
using EzXML: Node, readxml, attributes, elements
using Dates
using Spec
import ..Ohio

const OHIODIR = joinpath(dirname(pathof(Ohio)), "..")
const DATADIR = joinpath(OHIODIR, "data")
const FIGURESDIR = joinpath(OHIODIR, "figures")

export parseohio, expatient, glucoselevels, carblevels

nm(x) = Symbol(x.name)

"Parse the content of an attribute, depending on its name"
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

"Parse XML to NamedTuple"
function parseohio(x::Node)
  nms = map(nm, attributes(x))
  nm_ = nm(x)
  attrs = map(parseattr, attributes(x))
  elements_ = elements(x)
  if isempty(elements_)
    NamedTuple{(:nm, nms...,)}((nm_, attrs...,))
  else
    vals = map(parseohio, elements_)
    NamedTuple{(:nm, nms..., :vals)}((nm_, attrs..., vals))
  end
end


"Extract a sequence of float data from raw values"
function glucoselevels(patientdata)
  glucosedata = patientdata.vals[1].vals
  glucosey =  (x -> x.value).(glucosedata)
end

"Extract a sequence of float data from raw values"
function carblevels(patientdata)
  carbdata = patientdata.vals[6].vals
  carby =  (x -> x.carbs).(carbdata)
end


"Example data from patient"
function expatient(; fn = joinpath(DATADIR, "train", "588-ws-training.xml"))
  doc = readxml(fn)
  parseohio(doc.root)
end

end