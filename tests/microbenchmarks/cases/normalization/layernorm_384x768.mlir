// Layer norm over the last dimension written as generics (mean, variance,
// normalize, scale and shift by constant gamma/beta), 384x768 f32.

util.global private @gamma = #util.byte_pattern<63> : tensor<768xf32>
util.global private @beta = #util.byte_pattern<63> : tensor<768xf32>

func.func @layernorm_384x768(%x: tensor<384x768xf32>) -> tensor<384x768xf32> {
  %gamma = util.global.load @gamma : tensor<768xf32>
  %beta = util.global.load @beta : tensor<768xf32>
  %zero = arith.constant 0.0 : f32
  %er = tensor.empty() : tensor<384xf32>
  %fz = linalg.fill ins(%zero : f32) outs(%er : tensor<384xf32>) -> tensor<384xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<384x768xf32>) outs(%fz : tensor<384xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %a = arith.addf %in, %acc : f32
    linalg.yield %a : f32
  } -> tensor<384xf32>
  %mean = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%sum : tensor<384xf32>) outs(%er : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.constant 768.0 : f32
    %d = arith.divf %in, %n : f32
    linalg.yield %d : f32
  } -> tensor<384xf32>
  %var_sum = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%x, %mean : tensor<384x768xf32>, tensor<384xf32>) outs(%fz : tensor<384xf32>) {
  ^bb0(%in: f32, %m: f32, %acc: f32):
    %c = arith.subf %in, %m : f32
    %sq = arith.mulf %c, %c : f32
    %a = arith.addf %sq, %acc : f32
    linalg.yield %a : f32
  } -> tensor<384xf32>
  %e = tensor.empty() : tensor<384x768xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%x, %mean, %var_sum, %gamma, %beta : tensor<384x768xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768xf32>, tensor<768xf32>)
      outs(%e : tensor<384x768xf32>) {
  ^bb0(%in: f32, %m: f32, %vs: f32, %g: f32, %b: f32, %out: f32):
    %n = arith.constant 768.0 : f32
    %eps = arith.constant 1.0e-05 : f32
    %var = arith.divf %vs, %n : f32
    %ve = arith.addf %var, %eps : f32
    %rs = math.rsqrt %ve : f32
    %c = arith.subf %in, %m : f32
    %nrm = arith.mulf %c, %rs : f32
    %sc = arith.mulf %nrm, %g : f32
    %sh = arith.addf %sc, %b : f32
    linalg.yield %sh : f32
  } -> tensor<384x768xf32>
  return %r : tensor<384x768xf32>
}
