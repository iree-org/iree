func.func @scaled_matmul_1x1() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<1x1x32xf32>
  %lhs_scales = util.unfoldable_constant dense<126> : tensor<1x1xi8>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<1x1x32xf32>
  %rhs_scales = util.unfoldable_constant dense<126> : tensor<1x1xi8>
  %A = arith.truncf %lhs : tensor<1x1x32xf32> to tensor<1x1x32xf4E2M1FN>
  %A_scales = arith.bitcast %lhs_scales : tensor<1x1xi8> to tensor<1x1xf8E8M0FNU>
  %B = arith.truncf %rhs : tensor<1x1x32xf32> to tensor<1x1x32xf4E2M1FN>
  %B_scales = arith.bitcast %rhs_scales : tensor<1x1xi8> to tensor<1x1xf8E8M0FNU>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %C = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %D = linalg.generic {
    indexing_maps = [affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>, affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>, affine_map<(M, N, Ko, Kb) -> (M, Ko)>, affine_map<(M, N, Ko, Kb) -> (N, Ko)>, affine_map<(M, N, Ko, Kb) -> (M, N)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<1x1x32xf4E2M1FN>, tensor<1x1x32xf4E2M1FN>, tensor<1x1xf8E8M0FNU>, tensor<1x1xf8E8M0FNU>) outs(%C : tensor<1x1xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1x1xf32>
  check.expect_almost_eq_const(%D, dense<8.0> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

func.func @scaled_matmul_1024x1024() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<1024x32x32xf32>
  %lhs_scales = util.unfoldable_constant dense<127> : tensor<1024x32xi8>
  %rhs = util.unfoldable_constant dense<0.5> : tensor<1024x32x32xf32>
  %rhs_scales = util.unfoldable_constant dense<127> : tensor<1024x32xi8>
  %A = arith.truncf %lhs : tensor<1024x32x32xf32> to tensor<1024x32x32xf4E2M1FN>
  %A_scales = arith.bitcast %lhs_scales : tensor<1024x32xi8> to tensor<1024x32xf8E8M0FNU>
  %B = arith.truncf %rhs : tensor<1024x32x32xf32> to tensor<1024x32x32xf4E2M1FN>
  %B_scales = arith.bitcast %rhs_scales : tensor<1024x32xi8> to tensor<1024x32xf8E8M0FNU>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1024x1024xf32>
  %C = linalg.fill ins(%cst : f32) outs(%empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %D = linalg.generic {
    indexing_maps = [affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>, affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>, affine_map<(M, N, Ko, Kb) -> (M, Ko)>, affine_map<(M, N, Ko, Kb) -> (N, Ko)>, affine_map<(M, N, Ko, Kb) -> (M, N)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<1024x32x32xf4E2M1FN>, tensor<1024x32x32xf4E2M1FN>, tensor<1024x32xf8E8M0FNU>, tensor<1024x32xf8E8M0FNU>) outs(%C : tensor<1024x1024xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1024x1024xf32>
  check.expect_almost_eq_const(%D, dense<512.0> : tensor<1024x1024xf32>) : tensor<1024x1024xf32>
  return
}

func.func @scaled_matmul_128x128() {
  %lhs = util.unfoldable_constant dense<2.0> : tensor<128x4x32xf32>
  %lhs_scales = util.unfoldable_constant dense<126> : tensor<128x4xi8>
  %rhs = util.unfoldable_constant dense<2.0> : tensor<128x4x32xf32>
  %rhs_scales = util.unfoldable_constant dense<126> : tensor<128x4xi8>
  %A = arith.truncf %lhs : tensor<128x4x32xf32> to tensor<128x4x32xf4E2M1FN>
  %A_scales = arith.bitcast %lhs_scales : tensor<128x4xi8> to tensor<128x4xf8E8M0FNU>
  %B = arith.truncf %rhs : tensor<128x4x32xf32> to tensor<128x4x32xf4E2M1FN>
  %B_scales = arith.bitcast %rhs_scales : tensor<128x4xi8> to tensor<128x4xf8E8M0FNU>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %C = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %D = linalg.generic {
    indexing_maps = [affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>, affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>, affine_map<(M, N, Ko, Kb) -> (M, Ko)>, affine_map<(M, N, Ko, Kb) -> (N, Ko)>, affine_map<(M, N, Ko, Kb) -> (M, N)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<128x4x32xf4E2M1FN>, tensor<128x4x32xf4E2M1FN>, tensor<128x4xf8E8M0FNU>, tensor<128x4xf8E8M0FNU>) outs(%C : tensor<128x128xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<128x128xf32>
  check.expect_almost_eq_const(%D, dense<128.0> : tensor<128x128xf32>) : tensor<128x128xf32>
  return
}
