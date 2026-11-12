// RMS norm over the last dimension written as generics, constant weight,
// 128x4096 f32.

util.global private @weight = #util.byte_pattern<63> : tensor<4096xf32>

func.func @rmsnorm_128x4096(%x: tensor<128x4096xf32>) -> tensor<128x4096xf32> {
  %weight = util.global.load @weight : tensor<4096xf32>
  %zero = arith.constant 0.0 : f32
  %er = tensor.empty() : tensor<128xf32>
  %fz = linalg.fill ins(%zero : f32) outs(%er : tensor<128xf32>) -> tensor<128xf32>
  %ss = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<128x4096xf32>) outs(%fz : tensor<128xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %sq = arith.mulf %in, %in : f32
    %a = arith.addf %sq, %acc : f32
    linalg.yield %a : f32
  } -> tensor<128xf32>
  %e = tensor.empty() : tensor<128x4096xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%x, %ss, %weight : tensor<128x4096xf32>, tensor<128xf32>, tensor<4096xf32>) outs(%e : tensor<128x4096xf32>) {
  ^bb0(%in: f32, %s: f32, %w: f32, %out: f32):
    %n = arith.constant 4096.0 : f32
    %eps = arith.constant 1.0e-06 : f32
    %ms = arith.divf %s, %n : f32
    %me = arith.addf %ms, %eps : f32
    %rs = math.rsqrt %me : f32
    %nrm = arith.mulf %in, %rs : f32
    %y = arith.mulf %nrm, %w : f32
    linalg.yield %y : f32
  } -> tensor<128x4096xf32>
  return %r : tensor<128x4096xf32>
}
