// Max over the innermost dimension, 1024x1024 f32 -> 1024.

func.func @max_inner_1024x1024(%x: tensor<1024x1024xf32>) -> tensor<1024xf32> {
  %init = arith.constant 0xFF800000 : f32
  %e = tensor.empty() : tensor<1024xf32>
  %f = linalg.fill ins(%init : f32) outs(%e : tensor<1024xf32>) -> tensor<1024xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<1024x1024xf32>) outs(%f : tensor<1024xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %s = arith.maximumf %in, %acc : f32
    linalg.yield %s : f32
  } -> tensor<1024xf32>
  return %r : tensor<1024xf32>
}
