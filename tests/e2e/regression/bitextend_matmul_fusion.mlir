// Bit-extend producers feeding a matmul, as in the dequantization of
// weights or activations. On CPU targets with the mmt4d ukernel, dispatch
// creation fuses each producer into the matmul dispatch.

#map = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>
#row = affine_map<(d0, d1) -> (d0)>

func.func @dequant_i8_rhs_matmul_m1() {
  %a = util.unfoldable_constant dense<[[-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0]]> : tensor<1x8xf32>
  %w = util.unfoldable_constant dense<[[-6, 1, -5, 2], [-4, 3, -3, 4], [-2, 5, -1, 6], [0, -6, 1, -5], [2, -4, 3, -3], [4, -2, 5, -1], [6, 0, -6, 1], [-5, 2, -4, 3]]> : tensor<8x4xi8>
  %s = util.unfoldable_constant dense<[0.5, 2.0, 0.25, 1.0]> : tensor<4xf32>
  %e = tensor.empty() : tensor<8x4xf32>
  %x = linalg.generic {indexing_maps = [#map, #col, #map], iterator_types = ["parallel", "parallel"]} ins(%w, %s : tensor<8x4xi8>, tensor<4xf32>) outs(%e : tensor<8x4xf32>) {
  ^bb0(%in: i8, %scale: f32, %out: f32):
    %c = arith.sitofp %in : i8 to f32
    %d = arith.mulf %c, %scale : f32
    linalg.yield %d : f32
  } -> tensor<8x4xf32>
  %z = arith.constant 0.0 : f32
  %o = tensor.empty() : tensor<1x4xf32>
  %f = linalg.fill ins(%z : f32) outs(%o : tensor<1x4xf32>) -> tensor<1x4xf32>
  %r = linalg.matmul ins(%a, %x : tensor<1x8xf32>, tensor<8x4xf32>) outs(%f : tensor<1x4xf32>) -> tensor<1x4xf32>
  check.expect_almost_eq_const(%r, dense<[[3.0, -30.0, 4.0, -18.0]]> : tensor<1x4xf32>) : tensor<1x4xf32>
  return
}

func.func @dequant_i8_rhs_matmul_dynamic_m() {
  %c0 = arith.constant 0 : index
  %a = flow.tensor.dynamic_constant dense<[[-3.0, 0.0, 3.0, -1.0, 2.0, -2.0, 1.0, -3.0], [0.0, 3.0, -1.0, 2.0, -2.0, 1.0, -3.0, 0.0], [3.0, -1.0, 2.0, -2.0, 1.0, -3.0, 0.0, 3.0]]> : tensor<3x8xf32> -> tensor<?x8xf32>
  %m = tensor.dim %a, %c0 : tensor<?x8xf32>
  %w = util.unfoldable_constant dense<[[-6, 1, -5, 2], [-4, 3, -3, 4], [-2, 5, -1, 6], [0, -6, 1, -5], [2, -4, 3, -3], [4, -2, 5, -1], [6, 0, -6, 1], [-5, 2, -4, 3]]> : tensor<8x4xi8>
  %s = util.unfoldable_constant dense<[0.5, 2.0, 0.25, 1.0]> : tensor<4xf32>
  %e = tensor.empty() : tensor<8x4xf32>
  %x = linalg.generic {indexing_maps = [#map, #col, #map], iterator_types = ["parallel", "parallel"]} ins(%w, %s : tensor<8x4xi8>, tensor<4xf32>) outs(%e : tensor<8x4xf32>) {
  ^bb0(%in: i8, %scale: f32, %out: f32):
    %c = arith.sitofp %in : i8 to f32
    %d = arith.mulf %c, %scale : f32
    linalg.yield %d : f32
  } -> tensor<8x4xf32>
  %z = arith.constant 0.0 : f32
  %o = tensor.empty(%m) : tensor<?x4xf32>
  %f = linalg.fill ins(%z : f32) outs(%o : tensor<?x4xf32>) -> tensor<?x4xf32>
  %r = linalg.matmul ins(%a, %x : tensor<?x8xf32>, tensor<8x4xf32>) outs(%f : tensor<?x4xf32>) -> tensor<?x4xf32>
  %rb = util.optimization_barrier %r : tensor<?x4xf32>
  %rs = tensor.cast %rb : tensor<?x4xf32> to tensor<3x4xf32>
  check.expect_almost_eq_const(%rs, dense<[[14.5, 16.0, 3.25, 5.0], [-14.0, -4.0, 2.75, -2.0], [-21.5, 60.0, -10.0, 33.0]]> : tensor<3x4xf32>) : tensor<3x4xf32>
  return
}

func.func @dequant_i8_lhs_matmul() {
  %a = util.unfoldable_constant dense<[[-4, -3, -2, -1, 0, 1, 2, 3], [4, -4, -3, -2, -1, 0, 1, 2], [3, 4, -4, -3, -2, -1, 0, 1]]> : tensor<3x8xi8>
  %s = util.unfoldable_constant dense<[1.0, 0.5, 2.0]> : tensor<3xf32>
  %b = util.unfoldable_constant dense<[[-2.0, -1.0, 0.0, 1.0], [2.0, 3.0, -2.0, -1.0], [0.0, 1.0, 2.0, 3.0], [-2.0, -1.0, 0.0, 1.0], [2.0, 3.0, -2.0, -1.0], [0.0, 1.0, 2.0, 3.0], [-2.0, -1.0, 0.0, 1.0], [2.0, 3.0, -2.0, -1.0]]> : tensor<8x4xf32>
  %e = tensor.empty() : tensor<3x8xf32>
  %x = linalg.generic {indexing_maps = [#map, #row, #map], iterator_types = ["parallel", "parallel"]} ins(%a, %s : tensor<3x8xi8>, tensor<3xf32>) outs(%e : tensor<3x8xf32>) {
  ^bb0(%in: i8, %scale: f32, %out: f32):
    %c = arith.sitofp %in : i8 to f32
    %d = arith.mulf %c, %scale : f32
    linalg.yield %d : f32
  } -> tensor<3x8xf32>
  %z = arith.constant 0.0 : f32
  %o = tensor.empty() : tensor<3x4xf32>
  %f = linalg.fill ins(%z : f32) outs(%o : tensor<3x4xf32>) -> tensor<3x4xf32>
  %r = linalg.matmul ins(%x, %b : tensor<3x8xf32>, tensor<8x4xf32>) outs(%f : tensor<3x4xf32>) -> tensor<3x4xf32>
  check.expect_almost_eq_const(%r, dense<[[6.0, 2.0, -2.0, -6.0], [-6.0, -7.5, 0.0, -1.5], [12.0, 8.0, -32.0, -36.0]]> : tensor<3x4xf32>) : tensor<3x4xf32>
  return
}

func.func @scaled_extf_f16_rhs_matmul() {
  %a = util.unfoldable_constant dense<[[-3.0, 0.0, 3.0, -1.0, 2.0, -2.0, 1.0, -3.0], [0.0, 3.0, -1.0, 2.0, -2.0, 1.0, -3.0, 0.0], [3.0, -1.0, 2.0, -2.0, 1.0, -3.0, 0.0, 3.0]]> : tensor<3x8xf32>
  %w = util.unfoldable_constant dense<[[-6.0, 1.0, -5.0, 2.0], [-4.0, 3.0, -3.0, 4.0], [-2.0, 5.0, -1.0, 6.0], [0.0, -6.0, 1.0, -5.0], [2.0, -4.0, 3.0, -3.0], [4.0, -2.0, 5.0, -1.0], [6.0, 0.0, -6.0, 1.0], [-5.0, 2.0, -4.0, 3.0]]> : tensor<8x4xf16>
  %s = util.unfoldable_constant dense<[0.5, 2.0, 0.25, 1.0]> : tensor<4xf32>
  %e = tensor.empty() : tensor<8x4xf32>
  %x = linalg.generic {indexing_maps = [#map, #col, #map], iterator_types = ["parallel", "parallel"]} ins(%w, %s : tensor<8x4xf16>, tensor<4xf32>) outs(%e : tensor<8x4xf32>) {
  ^bb0(%in: f16, %scale: f32, %out: f32):
    %c = arith.extf %in : f16 to f32
    %d = arith.mulf %c, %scale : f32
    linalg.yield %d : f32
  } -> tensor<8x4xf32>
  %z = arith.constant 0.0 : f32
  %o = tensor.empty() : tensor<3x4xf32>
  %f = linalg.fill ins(%z : f32) outs(%o : tensor<3x4xf32>) -> tensor<3x4xf32>
  %r = linalg.matmul ins(%a, %x : tensor<3x8xf32>, tensor<8x4xf32>) outs(%f : tensor<3x4xf32>) -> tensor<3x4xf32>
  check.expect_almost_eq_const(%r, dense<[[14.5, 16.0, 3.25, 5.0], [-14.0, -4.0, 2.75, -2.0], [-21.5, 60.0, -10.0, 33.0]]> : tensor<3x4xf32>) : tensor<3x4xf32>
  return
}
