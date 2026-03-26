// Test RDNA3 WMMA f16/bf16 accumulator correctness for matmul.
//
// Regression test for a bug where WMMAR3_F16_16x16x16_F16 accumulator layout
// on gfx1100 (RDNA3) was incorrect: outer={16,1} claimed 16 valid elements per
// lane but v_wmma_f16_16x16x16_f16 only produces 8 valid results at even
// indices. The fix changes the layout to outer={8,1}, thread={2,16}.
// See amdgpu.wmma op docs in AMDGPUOps.td for the gfx11 subwordOffset layout.
//
// Uses LHS * I = LHS (matmul with identity) so each output element has a
// unique expected value. LHS values are unique per element (row and column
// dependent) to catch both row-swap and column-permutation bugs.
//
// The optimization_barrier prevents fusion of data generation with the matmul,
// ensuring the matmul gets its own dispatch and triggers the WMMA codegen path
// via vector distribution on gfx1100.

func.func @wmma_matmul_f16_identity() {
  // Build LHS with unique values: element [i,j] = (i * 128 + j) / 16384.
  // Range [0, ~1.0] stays within f16 precision.
  %empty_lhs = tensor.empty() : tensor<128x128xf16>
  %c128 = arith.constant 128.0 : f32
  %c16384 = arith.constant 16384.0 : f32
  %lhs_gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%empty_lhs : tensor<128x128xf16>) {
  ^bb0(%out: f16):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    %i_f32 = arith.sitofp %i_i32 : i32 to f32
    %j_f32 = arith.sitofp %j_i32 : i32 to f32
    %row = arith.mulf %i_f32, %c128 : f32
    %linear = arith.addf %row, %j_f32 : f32
    %val_f32 = arith.divf %linear, %c16384 : f32
    %val = arith.truncf %val_f32 : f32 to f16
    linalg.yield %val : f16
  } -> tensor<128x128xf16>

  // Build 128x128 identity matrix.
  %empty_rhs = tensor.empty() : tensor<128x128xf16>
  %zero_f16 = arith.constant 0.0 : f16
  %one_f16 = arith.constant 1.0 : f16
  %rhs_gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%empty_rhs : tensor<128x128xf16>) {
  ^bb0(%out: f16):
    %r = linalg.index 0 : index
    %c = linalg.index 1 : index
    %eq = arith.cmpi eq, %r, %c : index
    %val = arith.select %eq, %one_f16, %zero_f16 : f16
    linalg.yield %val : f16
  } -> tensor<128x128xf16>

  // Prevent fusion of data generation with the matmul so the matmul gets
  // its own dispatch and the WMMA codegen path is selected.
  %lhs = util.optimization_barrier %lhs_gen : tensor<128x128xf16>
  %rhs = util.optimization_barrier %rhs_gen : tensor<128x128xf16>

  // Matmul: LHS * I = LHS
  %cst = arith.constant 0.0 : f16
  %empty_out = tensor.empty() : tensor<128x128xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%empty_out : tensor<128x128xf16>) -> tensor<128x128xf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf16>, tensor<128x128xf16>)
    outs(%fill : tensor<128x128xf16>) -> tensor<128x128xf16>

  check.expect_almost_eq(%result, %lhs) : tensor<128x128xf16>
  return
}

func.func @wmma_matmul_bf16_identity() {
  // Same test for bf16 to cover WMMAR3_BF16_16x16x16_BF16.
  %empty_lhs = tensor.empty() : tensor<128x128xbf16>
  %c128 = arith.constant 128.0 : f32
  %c16384 = arith.constant 16384.0 : f32
  %lhs_gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%empty_lhs : tensor<128x128xbf16>) {
  ^bb0(%out: bf16):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    %i_f32 = arith.sitofp %i_i32 : i32 to f32
    %j_f32 = arith.sitofp %j_i32 : i32 to f32
    %row = arith.mulf %i_f32, %c128 : f32
    %linear = arith.addf %row, %j_f32 : f32
    %val_f32 = arith.divf %linear, %c16384 : f32
    %val = arith.truncf %val_f32 : f32 to bf16
    linalg.yield %val : bf16
  } -> tensor<128x128xbf16>

  %empty_rhs = tensor.empty() : tensor<128x128xbf16>
  %zero_bf16 = arith.constant 0.0 : bf16
  %one_bf16 = arith.constant 1.0 : bf16
  %rhs_gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%empty_rhs : tensor<128x128xbf16>) {
  ^bb0(%out: bf16):
    %r = linalg.index 0 : index
    %c = linalg.index 1 : index
    %eq = arith.cmpi eq, %r, %c : index
    %val = arith.select %eq, %one_bf16, %zero_bf16 : bf16
    linalg.yield %val : bf16
  } -> tensor<128x128xbf16>

  %lhs = util.optimization_barrier %lhs_gen : tensor<128x128xbf16>
  %rhs = util.optimization_barrier %rhs_gen : tensor<128x128xbf16>

  %cst = arith.constant 0.0 : bf16
  %empty_out = tensor.empty() : tensor<128x128xbf16>
  %fill = linalg.fill ins(%cst : bf16) outs(%empty_out : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<128x128xbf16>, tensor<128x128xbf16>)
    outs(%fill : tensor<128x128xbf16>) -> tensor<128x128xbf16>

  check.expect_almost_eq(%result, %lhs) : tensor<128x128xbf16>
  return
}
