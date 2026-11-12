// Average pool 3x3 stride 1 as pooling_nhwc_sum followed by a divide, NHWC,
// 1x30x30x128 -> 1x28x28x128 f32.

func.func @avgpool_nhwc_3x3_s1(%x: tensor<1x30x30x128xf32>) -> tensor<1x28x28x128xf32> {
  %zero = arith.constant 0.0 : f32
  %win = tensor.empty() : tensor<3x3xf32>
  %e = tensor.empty() : tensor<1x28x28x128xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %s = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%x, %win : tensor<1x30x30x128xf32>, tensor<3x3xf32>) outs(%f : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %e1 = tensor.empty() : tensor<1x28x28x128xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>, affine_map<(n, h, w, c) -> (n, h, w, c)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%s : tensor<1x28x28x128xf32>) outs(%e1 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %nine = arith.constant 9.0 : f32
    %d = arith.divf %in, %nine : f32
    linalg.yield %d : f32
  } -> tensor<1x28x28x128xf32>
  return %r : tensor<1x28x28x128xf32>
}
