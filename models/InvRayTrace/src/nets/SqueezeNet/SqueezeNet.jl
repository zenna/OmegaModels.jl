"SqueezeNet"
module SqueezeNet
using Statistics 
using ONNX
using Flux

export squeezenet
@show pwd()
# This is shaky, fixme
const weights_path = joinpath("SqueezeNet", "weights.bson")
const weights = ONNX.load_weights(weights_path)

Mul(a,b,c) = b .* reshape(c, (1,1,size(c)[a],1)) 
Add(axis, A ,B) = A .+ reshape(B, (1,1,size(B)[1],1)) 
const c_1 = MeanPool((13, 13), pad=(0, 0, 0, 0), stride=(13, 13))
const c_2 = CrossCor(weights["squeezenet0_conv25_weight"], weights["squeezenet0_conv25_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_3 = CrossCor(weights["squeezenet0_conv23_weight"], weights["squeezenet0_conv23_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_4 = CrossCor(weights["squeezenet0_conv22_weight"], weights["squeezenet0_conv22_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_5 = CrossCor(weights["squeezenet0_conv20_weight"], weights["squeezenet0_conv20_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_6 = CrossCor(weights["squeezenet0_conv19_weight"], weights["squeezenet0_conv19_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_7 = CrossCor(weights["squeezenet0_conv17_weight"], weights["squeezenet0_conv17_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_8 = CrossCor(weights["squeezenet0_conv16_weight"], weights["squeezenet0_conv16_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_9 = CrossCor(weights["squeezenet0_conv14_weight"], weights["squeezenet0_conv14_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_10 = CrossCor(weights["squeezenet0_conv13_weight"], weights["squeezenet0_conv13_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_11 = MaxPool((3, 3), pad=(0, 0, 0, 0), stride=(2, 2))
const c_12 = CrossCor(weights["squeezenet0_conv11_weight"], weights["squeezenet0_conv11_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_13 = CrossCor(weights["squeezenet0_conv10_weight"], weights["squeezenet0_conv10_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_14 = CrossCor(weights["squeezenet0_conv8_weight"], weights["squeezenet0_conv8_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_15 = CrossCor(weights["squeezenet0_conv7_weight"], weights["squeezenet0_conv7_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_16 = MaxPool((3, 3), pad=(0, 0, 0, 0), stride=(2, 2))
const c_17 = CrossCor(weights["squeezenet0_conv5_weight"], weights["squeezenet0_conv5_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_18 = CrossCor(weights["squeezenet0_conv4_weight"], weights["squeezenet0_conv4_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_19 = CrossCor(weights["squeezenet0_conv2_weight"], weights["squeezenet0_conv2_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_20 = CrossCor(weights["squeezenet0_conv1_weight"], weights["squeezenet0_conv1_bias"], stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_21 = MaxPool((3, 3), pad=(0, 0, 0, 0), stride=(2, 2))
const c_22 = CrossCor(weights["squeezenet0_conv0_weight"], weights["squeezenet0_conv0_bias"], stride=(2, 2), pad=(0, 0, 0, 0), dilation=(1, 1))
const c_23 = CrossCor(weights["squeezenet0_conv3_weight"], weights["squeezenet0_conv3_bias"], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_24 = CrossCor(weights["squeezenet0_conv6_weight"], weights["squeezenet0_conv6_bias"], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_25 = CrossCor(weights["squeezenet0_conv9_weight"], weights["squeezenet0_conv9_bias"], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_26 = CrossCor(weights["squeezenet0_conv12_weight"], weights["squeezenet0_conv12_bias"], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_27 = CrossCor(weights["squeezenet0_conv15_weight"], weights["squeezenet0_conv15_bias"], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_28 = CrossCor(weights["squeezenet0_conv18_weight"], weights["squeezenet0_conv18_bias"], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_29 = CrossCor(weights["squeezenet0_conv21_weight"], weights["squeezenet0_conv21_bias"], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_30 = CrossCor(weights["squeezenet0_conv24_weight"], weights["squeezenet0_conv24_bias"], stride=(1, 1), pad=(1, 1, 1, 1), dilation=(1, 1))
const c_31 = broadcast(Int64, Tuple(reverse(weights["reshape_attr_tensor118"])))

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