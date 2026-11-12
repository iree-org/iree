// Scaled dot-product attention written as frontends emit it: Q x K^T batch
// matmul, scale, softmax, x V batch matmul. 12 heads, sequence 384, head dim 64, f32.

func.func @attention_decomposed_12x384x64(%q: tensor<12x384x64xf32>, %k: tensor<12x384x64xf32>, %v: tensor<12x384x64xf32>) -> tensor<12x384x64xf32> {
  %zero = arith.constant 0.0 : f32
  %es = tensor.empty() : tensor<12x384x384xf32>
  %fs = linalg.fill ins(%zero : f32) outs(%es : tensor<12x384x384xf32>) -> tensor<12x384x384xf32>
  %s = linalg.batch_matmul indexing_maps = [
          affine_map<(b, m, n, k) -> (b, m, k)>,
          affine_map<(b, m, n, k) -> (b, n, k)>,
          affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%q, %k : tensor<12x384x64xf32>, tensor<12x384x64xf32>) outs(%fs : tensor<12x384x384xf32>) -> tensor<12x384x384xf32>
  %e1 = tensor.empty() : tensor<12x384x384xf32>
  %ss = linalg.generic {
      indexing_maps = [affine_map<(b, i, j) -> (b, i, j)>, affine_map<(b, i, j) -> (b, i, j)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%s : tensor<12x384x384xf32>) outs(%e1 : tensor<12x384x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.constant 0.125 : f32
    %m = arith.mulf %in, %c : f32
    linalg.yield %m : f32
  } -> tensor<12x384x384xf32>
  %e2 = tensor.empty() : tensor<12x384x384xf32>
  %p = linalg.softmax dimension(2) ins(%ss : tensor<12x384x384xf32>) outs(%e2 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32>
  %eo = tensor.empty() : tensor<12x384x64xf32>
  %fo = linalg.fill ins(%zero : f32) outs(%eo : tensor<12x384x64xf32>) -> tensor<12x384x64xf32>
  %o = linalg.batch_matmul ins(%p, %v : tensor<12x384x384xf32>, tensor<12x384x64xf32>) outs(%fo : tensor<12x384x64xf32>) -> tensor<12x384x64xf32>
  return %o : tensor<12x384x64xf32>
}
