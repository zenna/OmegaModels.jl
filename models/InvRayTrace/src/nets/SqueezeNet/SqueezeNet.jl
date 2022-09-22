"SqueezeNet"
module SqueezeNet_
using Statistics 
using Flux
using Metalhead
export squeezenet

# This is shaky, fixme
const weights = Flux.params(SqueezeNet(pretrain = true))

Mul(a,b,c) = b .* reshape(c, (1,1,size(c)[a],1)) 
Add(axis, A ,B) = A .+ reshape(B, (1,1,size(B)[1],1)) 
const c_1 = MeanPool((13, 13), pad=(0, 0, 0, 0), stride=(13, 13))
const c_2 = CrossCor(weights[51], weights[52], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_3 = CrossCor(weights[47], weights[48], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_4 = CrossCor(weights[45], weights[46], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_5 = CrossCor(weights[41], weights[42], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_6 = CrossCor(weights[39], weights[40], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_7 = CrossCor(weights[35], weights[36], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_8 = CrossCor(weights[33], weights[34], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_9 = CrossCor(weights[29], weights[30], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_10 = CrossCor(weights[27], weights[28], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_11 = MaxPool((3, 3), pad=(0, 0, 0, 0), stride=(2, 2))
const c_12 = CrossCor(weights[23], weights[24], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_13 = CrossCor(weights[21], weights[22], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_14 = CrossCor(weights[17], weights[18], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_15 = CrossCor(weights[15], weights[16], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_16 = MaxPool((3, 3), pad=(0, 0, 0, 0), stride=(2, 2))
const c_17 = CrossCor(weights[11], weights[12], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_18 = CrossCor(weights[9], weights[10], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_19 = CrossCor(weights[5], weights[6], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_20 = CrossCor(weights[3], weights[4], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_21 = MaxPool((3, 3), pad=(0, 0, 0, 0), stride=(2, 2))
const c_22 = CrossCor(weights[1], weights[2], stride=(2, 2), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_23 = CrossCor(weights[7], weights[8], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_24 = CrossCor(weights[13], weights[14], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_25 = CrossCor(weights[19], weights[20], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_26 = CrossCor(weights[25], weights[26], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_27 = CrossCor(weights[31], weights[32], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_28 = CrossCor(weights[37], weights[38], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_29 = CrossCor(weights[43], weights[44], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_30 = CrossCor(weights[49], weights[50], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
# const c_31 = broadcast(Int64, Tuple(reverse(weights["reshape_attr_tensor118"])))

"Takes as input image of size (244, 224, 3, 1)"
function squeezenet(x_32)
    edge_33 = relu.(c_20(c_21(relu.(c_22(x_32)))))
    edge_34 = relu.(c_18(cat(relu.(c_19(edge_33)), relu.(c_23(edge_33)), dims = 3)))
    edge_35 = relu.(c_15(c_16(cat(relu.(c_17(edge_34)), relu.(c_24(edge_34)), dims = 3))))
    edge_36 = relu.(c_13(cat(relu.(c_14(edge_35)), relu.(c_25(edge_35)), dims = 3)))
    edge_37 = relu.(c_10(c_11(cat(relu.(c_12(edge_36)), relu.(c_26(edge_36)), dims = 3))))
    edge_38 = relu.(c_8(cat(relu.(c_9(edge_37)), relu.(c_27(edge_37)), dims = 3)))
    (x_32, edge_33, edge_33, edge_35, edge_36, edge_37, edge_38)
    # edge_39 = relu.(c_6(cat(relu.(c_7(edge_38)), relu.(c_28(edge_38)), dims = 3)))
    # edge_40 = relu.(c_4(cat(relu.(c_5(edge_39)), relu.(c_29(edge_39)), dims = 3)))
    # c_1(relu.(c_2(identity(cat(relu.(c_3(edge_40)), relu.(c_30(edge_40)), dims = 3)))))
end
end