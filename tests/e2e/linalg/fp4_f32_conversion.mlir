func.func @f4E2M1FN_scaled_to_f32() {
  // [0, 1.0, 2.0, 0.5] : f4E2M1FN
  %input = util.unfoldable_constant dense<[0, 2, 4, 1]> : tensor<4xi8>
  %scale = util.unfoldable_constant dense<2.0> : tensor<4xf8E8M0FNU>
  %init0 = tensor.empty() : tensor<4xf32>
  %res = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%input, %scale : tensor<4xi8>, tensor<4xf8E8M0FNU>) outs(%init0 : tensor<4xf32>) {
  ^bb0(%in: i8, %s: f8E8M0FNU, %out: f32):
    %0 = arith.trunci %in : i8 to i4
    %1 = arith.bitcast %0 : i4 to f4E2M1FN
    %2 = arith.scaling_extf %1, %s : f4E2M1FN, f8E8M0FNU to f32
    linalg.yield %2 : f32
  } -> tensor<4xf32>

  check.expect_eq_const(%res, dense<[0.0, 2.0, 4.0, 1.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @f32_to_f4E2M1FN_scaled() {
  %input = util.unfoldable_constant dense<[0.0, 2.0, 4.0, 1.0]> : tensor<4xf32>
  %scale = util.unfoldable_constant dense<2.0> : tensor<4xf8E8M0FNU>
  %init0 = tensor.empty() : tensor<4xi8>
  %res = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%input, %scale : tensor<4xf32>, tensor<4xf8E8M0FNU>) outs(%init0 : tensor<4xi8>) {
  ^bb0(%in: f32, %s: f8E8M0FNU, %out: i8):
    %0 = arith.scaling_truncf %in, %s : f32, f8E8M0FNU to f4E2M1FN
    %1 = arith.bitcast %0 : f4E2M1FN to i4
    %2 = arith.extui %1 : i4 to i8
    linalg.yield %2 : i8
  } -> tensor<4xi8>

  // [0.0, 1.0, 2.0, 0.5] : tensor<4xf4E2M1FN>
  check.expect_eq_const(%res, dense<[0, 2, 4, 1]> : tensor<4xi8>) : tensor<4xi8>
  return
}
