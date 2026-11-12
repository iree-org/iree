// Elementwise add where one operand is read transposed (a + b^T), 1024x1024 f32.

func.func @add_transposed_1024x1024(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %e = tensor.empty() : tensor<1024x1024xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (j, i)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e : tensor<1024x1024xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<1024x1024xf32>
  return %r : tensor<1024x1024xf32>
}
