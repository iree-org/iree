// Reductions whose inner dim is not a multiple of the subgroup size.
//
// Before the kernel-config relaxation, single-row shapes whose inner dim is
// coprime to (workgroup * threadLoads) (e.g. 257 here) reached
// LLVMGPUVectorDistribute with a degenerate per-thread tile of size 1 and
// failed with 'op failed to distribute'. The current relaxation routes
// these to VectorDistribute with workgroup_size = [128, 1, 1],
// subgroup_size = 64, threadLoads = 4, partial_reduction = 512, and tail
// masking absorbing the (workgroup_size * threadLoads) - reductionSize
// overshoot.

// Single-row reduction with a prime inner dim (257). Used to fail compile.
func.func @reduction_inner_dim_unaligned_prime() {
  %in = util.unfoldable_constant dense<1.0> : tensor<1x257xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1xf32>) -> tensor<1xf32>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<1x257xf32>) outs(%fill : tensor<1xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<1xf32>
  check.expect_eq_const(%result, dense<257.0> : tensor<1xf32>) : tensor<1xf32>
  return
}

// Multi-row reduction with a composite non-aligned inner dim (300). Used to
// compile but with sub-optimal degenerate per-thread tiles; now uses full
// subgroup participation across the 128-thread workgroup.
func.func @reduction_inner_dim_unaligned_composite_multirow() {
  %in = util.unfoldable_constant dense<1.0> : tensor<8x300xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<8xf32>) -> tensor<8xf32>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<8x300xf32>) outs(%fill : tensor<8xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
  check.expect_eq_const(%result, dense<300.0> : tensor<8xf32>) : tensor<8xf32>
  return
}

// Same prime inner dim as `reduction_inner_dim_unaligned_prime`, but with a
// distinct value at the LAST position (256). A masking bug that drops the
// tail element, reads garbage past lastReductionDimSize, or double-counts
// would produce a sum != 356.0; the all-ones case alone could not detect
// any of these failure modes.
func.func @reduction_inner_dim_unaligned_prime_tail_marker() {
  %ones = util.unfoldable_constant dense<1.0> : tensor<1x257xf32>
  %marker = util.unfoldable_constant dense<100.0> : tensor<1x1xf32>
  %in = tensor.insert_slice %marker into %ones[0, 256] [1, 1] [1, 1]
      : tensor<1x1xf32> into tensor<1x257xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1xf32>) -> tensor<1xf32>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<1x257xf32>) outs(%fill : tensor<1xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<1xf32>
  // 256 leading ones + 100.0 at position 256 = 356.0
  check.expect_eq_const(%result, dense<356.0> : tensor<1xf32>) : tensor<1xf32>
  return
}
