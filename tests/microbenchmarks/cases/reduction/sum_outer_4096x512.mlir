// Sum over the outermost dimension, 4096x512 f32 -> 512.

func.func @sum_outer_4096x512(%x: tensor<4096x512xf32>) -> tensor<512xf32> {
  %init = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<512xf32>
  %f = linalg.fill ins(%init : f32) outs(%e : tensor<512xf32>) -> tensor<512xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%x : tensor<4096x512xf32>) outs(%f : tensor<512xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %s = arith.addf %in, %acc : f32
    linalg.yield %s : f32
  } -> tensor<512xf32>
  return %r : tensor<512xf32>
}
