// Element type conversion f32 -> f16, 4096x4096.

func.func @truncf_f32_to_f16_4096x4096(%x: tensor<4096x4096xf32>) -> tensor<4096x4096xf16> {
  %e = tensor.empty() : tensor<4096x4096xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<4096x4096xf32>) outs(%e : tensor<4096x4096xf16>) {
  ^bb0(%v: f32, %out: f16):
    %t = arith.truncf %v : f32 to f16
    linalg.yield %t : f16
  } -> tensor<4096x4096xf16>
  return %r : tensor<4096x4096xf16>
}
