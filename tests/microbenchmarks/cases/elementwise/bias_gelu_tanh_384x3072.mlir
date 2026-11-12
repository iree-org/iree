// Broadcast bias add followed by tanh-approximated GELU, written as two separate
// generics that should fuse, 384x3072 f32.

util.global private @bias = #util.byte_pattern<63> : tensor<3072xf32>

func.func @bias_gelu_tanh_384x3072(%x: tensor<384x3072xf32>) -> tensor<384x3072xf32> {
  %bias = util.global.load @bias : tensor<3072xf32>
  %e0 = tensor.empty() : tensor<384x3072xf32>
  %a = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%x, %bias : tensor<384x3072xf32>, tensor<3072xf32>) outs(%e0 : tensor<384x3072xf32>) {
  ^bb0(%v: f32, %b: f32, %out: f32):
    %s = arith.addf %v, %b : f32
    linalg.yield %s : f32
  } -> tensor<384x3072xf32>
  %e1 = tensor.empty() : tensor<384x3072xf32>
  %g = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<384x3072xf32>) outs(%e1 : tensor<384x3072xf32>) {
  ^bb0(%v: f32, %out: f32):
    %half = arith.constant 0.5 : f32
    %one = arith.constant 1.0 : f32
    %c0 = arith.constant 0.7978845608 : f32
    %c1 = arith.constant 0.044715 : f32
    %v2 = arith.mulf %v, %v : f32
    %v3 = arith.mulf %v2, %v : f32
    %t0 = arith.mulf %v3, %c1 : f32
    %t1 = arith.addf %v, %t0 : f32
    %t2 = arith.mulf %t1, %c0 : f32
    %t3 = math.tanh %t2 : f32
    %t4 = arith.addf %t3, %one : f32
    %t5 = arith.mulf %v, %half : f32
    %r = arith.mulf %t5, %t4 : f32
    linalg.yield %r : f32
  } -> tensor<384x3072xf32>
  return %g : tensor<384x3072xf32>
}
