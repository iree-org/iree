// ResNet head global average pool as pooling_nchw_sum with a 7x7 window plus a
// divide, 1x2048x7x7 -> 1x2048x1x1 f32.

func.func @global_avgpool_nchw_1x2048x7x7(%x: tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32> {
  %zero = arith.constant 0.0 : f32
  %win = tensor.empty() : tensor<7x7xf32>
  %e = tensor.empty() : tensor<1x2048x1x1xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
  %s = linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%x, %win : tensor<1x2048x7x7xf32>, tensor<7x7xf32>) outs(%f : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
  %e1 = tensor.empty() : tensor<1x2048x1x1xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%s : tensor<1x2048x1x1xf32>) outs(%e1 : tensor<1x2048x1x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.constant 49.0 : f32
    %d = arith.divf %in, %n : f32
    linalg.yield %d : f32
  } -> tensor<1x2048x1x1xf32>
  return %r : tensor<1x2048x1x1xf32>
}
