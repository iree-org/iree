// Residual add followed by relu, two activation inputs, 1x256x56x56 f32.

func.func @add_relu_1x256x56x56(%a: tensor<1x256x56x56xf32>, %b: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32> {
  %e = tensor.empty() : tensor<1x256x56x56xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (n, c, h, w)>,
                       affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%a, %b : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%e : tensor<1x256x56x56xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %zero = arith.constant 0.0 : f32
    %s = arith.addf %x, %y : f32
    %m = arith.maximumf %s, %zero : f32
    linalg.yield %m : f32
  } -> tensor<1x256x56x56xf32>
  return %r : tensor<1x256x56x56xf32>
}
