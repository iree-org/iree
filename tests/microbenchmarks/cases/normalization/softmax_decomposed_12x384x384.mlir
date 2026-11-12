// Softmax over the last dimension written as generics (max, exp(x - max), sum,
// divide), as frontends often emit it, 12x384x384 f32.

func.func @softmax_decomposed_12x384x384(%x: tensor<12x384x384xf32>) -> tensor<12x384x384xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %er = tensor.empty() : tensor<12x384xf32>
  %fm = linalg.fill ins(%ninf : f32) outs(%er : tensor<12x384xf32>) -> tensor<12x384xf32>
  %max = linalg.generic {
      indexing_maps = [affine_map<(b, i, j) -> (b, i, j)>, affine_map<(b, i, j) -> (b, i)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x : tensor<12x384x384xf32>) outs(%fm : tensor<12x384xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %m = arith.maximumf %in, %acc : f32
    linalg.yield %m : f32
  } -> tensor<12x384xf32>
  %e = tensor.empty() : tensor<12x384x384xf32>
  %ex = linalg.generic {
      indexing_maps = [affine_map<(b, i, j) -> (b, i, j)>, affine_map<(b, i, j) -> (b, i)>, affine_map<(b, i, j) -> (b, i, j)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x, %max : tensor<12x384x384xf32>, tensor<12x384xf32>) outs(%e : tensor<12x384x384xf32>) {
  ^bb0(%in: f32, %m: f32, %out: f32):
    %s = arith.subf %in, %m : f32
    %v = math.exp %s : f32
    linalg.yield %v : f32
  } -> tensor<12x384x384xf32>
  %fs = linalg.fill ins(%zero : f32) outs(%er : tensor<12x384xf32>) -> tensor<12x384xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(b, i, j) -> (b, i, j)>, affine_map<(b, i, j) -> (b, i)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%ex : tensor<12x384x384xf32>) outs(%fs : tensor<12x384xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %a = arith.addf %in, %acc : f32
    linalg.yield %a : f32
  } -> tensor<12x384xf32>
  %e2 = tensor.empty() : tensor<12x384x384xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(b, i, j) -> (b, i, j)>, affine_map<(b, i, j) -> (b, i)>, affine_map<(b, i, j) -> (b, i, j)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ex, %sum : tensor<12x384x384xf32>, tensor<12x384xf32>) outs(%e2 : tensor<12x384x384xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %d = arith.divf %in, %s : f32
    linalg.yield %d : f32
  } -> tensor<12x384x384xf32>
  return %r : tensor<12x384x384xf32>
}
