// QDQ conv 1x1 stride 2, NHWC/HWCF, 512 -> 1024 channels, 28x28 -> 14x14, bias.
// Per-tensor activation quantization (zero point -128), i8 weights with per-channel
// scale, requantized i8 output.

util.global private @weight = #util.byte_pattern<63> : tensor<1x1x512x1024xi8>
util.global private @weight_scale = #util.byte_pattern<63> : tensor<1024xf32>
util.global private @bias = #util.byte_pattern<63> : tensor<1024xf32>

func.func @conv2d_nhwc_1x1_s2_qdq(%x: tensor<1x28x28x512xf32>) -> tensor<1x14x14x1024xi8> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<1x28x28x512xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<1x28x28x512xf32>, f32, i8) outs(%e3 : tensor<1x28x28x512xi8>) -> tensor<1x28x28x512xi8>
  %e5 = tensor.empty() : tensor<1x28x28x512xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%q4, %c1, %c2 : tensor<1x28x28x512xi8>, f32, i8) outs(%e5 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %weight = util.global.load @weight : tensor<1x1x512x1024xi8>
  %weight_scale = util.global.load @weight_scale : tensor<1024xf32>
  %c7 = arith.constant dense<0> : tensor<1024xi8>
  %e8 = tensor.empty() : tensor<1x1x512x1024xf32>
  %wdq9 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%weight, %weight_scale, %c7 : tensor<1x1x512x1024xi8>, tensor<1024xf32>, tensor<1024xi8>) outs(%e8 : tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xf32>
  %bias = util.global.load @bias : tensor<1024xf32>
  %e10 = tensor.empty() : tensor<1x14x14x1024xf32>
  %b11 = linalg.broadcast ins(%bias : tensor<1024xf32>) outs(%e10 : tensor<1x14x14x1024xf32>) dimensions = [0, 1, 2]
  %conv12 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%dq6, %wdq9 : tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32>) outs(%b11 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %c13 = arith.constant 0.1 : f32
  %c14 = arith.constant -128 : i8
  %e15 = tensor.empty() : tensor<1x14x14x1024xi8>
  %q16 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%conv12, %c13, %c14 : tensor<1x14x14x1024xf32>, f32, i8) outs(%e15 : tensor<1x14x14x1024xi8>) -> tensor<1x14x14x1024xi8>
  return %q16 : tensor<1x14x14x1024xi8>
}
