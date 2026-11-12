// Sum over the spatial dimensions (global pooling as a reduction), 1x2048x7x7 f32.

func.func @sum_spatial_1x2048x7x7(%x: tensor<1x2048x7x7xf32>) -> tensor<1x2048xf32> {
  %init = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2048xf32>
  %f = linalg.fill ins(%init : f32) outs(%e : tensor<1x2048xf32>) -> tensor<1x2048xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%x : tensor<1x2048x7x7xf32>) outs(%f : tensor<1x2048xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %s = arith.addf %in, %acc : f32
    linalg.yield %s : f32
  } -> tensor<1x2048xf32>
  return %r : tensor<1x2048xf32>
}
