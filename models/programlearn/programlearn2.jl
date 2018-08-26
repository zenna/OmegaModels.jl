module ProgramLearn2

using Omega
using Distributions
using UnicodePlots

"Build a tiny PL"
abstract type TinyLang end
abstract type TLBinaryOp end

struct TLExpr <: TinyLang 
  op::TLBinaryOp
  left::TinyLang
  right::TinyLang
end

struct TLVal <: TinyLang
  val::Float64
end

struct TLVar <: TinyLang end

struct TLPlus <: TLBinaryOp end
struct TLMinus <: TLBinaryOp end
struct TLTimes <: TLBinaryOp end
struct TLDiv <: TLBinaryOp end

tl_eval(op::TLPlus, left, right) = left + right
tl_eval(op::TLMinus, left, right) = left - right
tl_eval(op::TLTimes, left, right) = left * right
tl_eval(op::TLDiv, left, right) = left / right

function tl_eval(expr::TLExpr, context::Number)
  left, right = [tl_eval(x, context) for x in [expr.left, expr.right]]
  tl_eval(expr.op, left, right)
end

tl_eval(val::TLVal, context) =  val.val
tl_eval(var::TLVar, context) =  context

"Sample random expressions"
function randexpr_(rng, weight=0.5)
  primitives = [TLPlus, TLMinus, TLTimes, TLDiv]
  op = rand(rng[@id], primitives)()
  left, right = randargs(rng[@id], weight)
  TLExpr(op, left, right)
end

"sample arguments for the binary expressions"
function randargs(rng, weight)
  args = []
  for i = 1:2
    p = rand(rng[@id][i])
    element = if p > weight
      TLVar()
    elseif p > weight / 2.0
      # randexpr_(rng[@id][i], weight * weight)
      randexpr_(rng[@id][i], weight * weight)
    else
      value = convert(Float64, rand(rng[@id][i], 1:20))
      TLVal(value)
    end
    push!(args, element)
  end
  args
end

xs = collect(0.00001:1.0:10.0)
fx_all(expr) = map(x->tl_eval(expr, x), xs)

function run(α = 100.0, n=10000)
  randexpr = ciid(randexpr_)
  fx = ciid(fx_all, randexpr)
  Omega.withkernel(Omega.kseα(α)) do
    rand(randexpr, fx ==ₛ sin.(xs), 10000; alg=SSMH)
  end
end


errors(exprs) = map(exprs) do expr
  sum((x->x^2).(fx_all(expr) - sin.(xs)))
end

function run_all()
  exprs = run(100)
  exprs |> errors |> lineplot
  exprs[2000:end] |> errors |> lineplot
end
