// Linear layer with fused epilogue: [384, 768] x [768, 3072] + bias, relu, f32.

util.global private @weight = #util.byte_pattern<63> : tensor<768x3072xf32>
util.global private @bias = #util.byte_pattern<63> : tensor<3072xf32>

func.func @linear_384x3072x768_bias_relu(%x: tensor<384x768xf32>) -> tensor<384x3072xf32> {
  %weight = util.global.load @weight : tensor<768x3072xf32>
  %bias = util.global.load @bias : tensor<3072xf32>
  %e1 = tensor.empty() : tensor<384x3072xf32>
  %b2 = linalg.broadcast ins(%bias : tensor<3072xf32>) outs(%e1 : tensor<384x3072xf32>) dimensions = [0]
  %mm3 = linalg.matmul ins(%x, %weight : tensor<384x768xf32>, tensor<768x3072xf32>) outs(%b2 : tensor<384x3072xf32>) -> tensor<384x3072xf32>
  %e4 = tensor.empty() : tensor<384x3072xf32>
  %g5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%mm3 : tensor<384x3072xf32>) outs(%e4 : tensor<384x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %r = arith.maximumf %in, %zero : f32
      linalg.yield %r : f32
  } -> tensor<384x3072xf32>
  return %g5 : tensor<384x3072xf32>
}
