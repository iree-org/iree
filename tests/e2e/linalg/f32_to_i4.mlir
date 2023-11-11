func.func @f32_to_i4_1d() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]> : tensor<8xf32>
  %init0 = tensor.empty() : tensor<8xi4>
  %res = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%input : tensor<8xf32>) outs(%init0 : tensor<8xi4>) {
  ^bb0(%in: f32, %out: i4):
    %2 = arith.fptoui %in : f32 to i32
    %3 = arith.trunci %2 : i32 to i4
    linalg.yield %3 : i4
  } -> tensor<8xi4>

  // TODO(#14996): Remove the signed extention and directly check with i4 types.
  %blocker = util.optimization_barrier %res : tensor<8xi4>
  %init1 = tensor.empty() : tensor<8xi8>
  %exti8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%blocker : tensor<8xi4>) outs(%init1 : tensor<8xi8>) {
  ^bb0(%in: i4, %out: i8):
    %2 = arith.extsi %in : i4 to i8
    linalg.yield %2 : i8
  } -> tensor<8xi8>

  check.expect_eq_const(%exti8, dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>) : tensor<8xi8>
  return
}
