// VDMFMA skinny GEMM (M=8) e2e tests for all supported element types.
// These exercise the smfmac sparse trick path on gfx942/gfx950.

// F16: VDMFMA_F32_8x16x64_F16
// M=8, N=64, K=128 (multi-N-tile to exercise subgroup=[1, 2+, 0]).
func.func @skinny_gemm_vdmfma_f16() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<8x128xf16>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<128x64xf16>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x64xf32>) -> tensor<8x64xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<8x128xf16>, tensor<128x64xf16>) outs(%fill : tensor<8x64xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in_0 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %2, %out : f32
    linalg.yield %3 : f32
  } -> tensor<8x64xf32>
  check.expect_eq_const(%result, dense<128.0> : tensor<8x64xf32>) : tensor<8x64xf32>
  return
}

// BF16: VDMFMA_F32_8x16x64_BF16
func.func @skinny_gemm_vdmfma_bf16() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<8x128xbf16>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<128x64xbf16>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x64xf32>) -> tensor<8x64xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<8x128xbf16>, tensor<128x64xbf16>) outs(%fill : tensor<8x64xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %0 = arith.extf %in : bf16 to f32
    %1 = arith.extf %in_0 : bf16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %2, %out : f32
    linalg.yield %3 : f32
  } -> tensor<8x64xf32>
  check.expect_eq_const(%result, dense<128.0> : tensor<8x64xf32>) : tensor<8x64xf32>
  return
}

// I8: VDMFMA_I32_8x16x128_I8
func.func @skinny_gemm_vdmfma_i8() {
  %lhs = util.unfoldable_constant dense<1> : tensor<8x128xi8>
  %rhs = util.unfoldable_constant dense<1> : tensor<128x64xi8>
  %cst = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<8x64xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x64xi32>) -> tensor<8x64xi32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<8x128xi8>, tensor<128x64xi8>) outs(%fill : tensor<8x64xi32>) {
  ^bb0(%in: i8, %in_0: i8, %out: i32):
    %0 = arith.extsi %in : i8 to i32
    %1 = arith.extsi %in_0 : i8 to i32
    %2 = arith.muli %0, %1 : i32
    %3 = arith.addi %2, %out : i32
    linalg.yield %3 : i32
  } -> tensor<8x64xi32>
  check.expect_eq_const(%result, dense<128> : tensor<8x64xi32>) : tensor<8x64xi32>
  return
}

// F8E4M3FNUZ: VDMFMA_F32_8x16x128_F8E4M3FNUZ
func.func @skinny_gemm_vdmfma_f8e4m3fnuz() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<8x128xf8E4M3FNUZ>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<128x64xf8E4M3FNUZ>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x64xf32>) -> tensor<8x64xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<8x128xf8E4M3FNUZ>, tensor<128x64xf8E4M3FNUZ>) outs(%fill : tensor<8x64xf32>) {
  ^bb0(%in: f8E4M3FNUZ, %in_0: f8E4M3FNUZ, %out: f32):
    %0 = arith.extf %in : f8E4M3FNUZ to f32
    %1 = arith.extf %in_0 : f8E4M3FNUZ to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %2, %out : f32
    linalg.yield %3 : f32
  } -> tensor<8x64xf32>
  check.expect_eq_const(%result, dense<128.0> : tensor<8x64xf32>) : tensor<8x64xf32>
  return
}

// F8E5M2FNUZ: VDMFMA_F32_8x16x128_F8E5M2FNUZ
func.func @skinny_gemm_vdmfma_f8e5m2fnuz() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<8x128xf8E5M2FNUZ>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<128x64xf8E5M2FNUZ>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x64xf32>) -> tensor<8x64xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<8x128xf8E5M2FNUZ>, tensor<128x64xf8E5M2FNUZ>) outs(%fill : tensor<8x64xf32>) {
  ^bb0(%in: f8E5M2FNUZ, %in_0: f8E5M2FNUZ, %out: f32):
    %0 = arith.extf %in : f8E5M2FNUZ to f32
    %1 = arith.extf %in_0 : f8E5M2FNUZ to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %2, %out : f32
    linalg.yield %3 : f32
  } -> tensor<8x64xf32>
  check.expect_eq_const(%result, dense<128.0> : tensor<8x64xf32>) : tensor<8x64xf32>
  return
}
