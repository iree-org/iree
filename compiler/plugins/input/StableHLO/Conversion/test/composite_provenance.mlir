// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline \
// RUN:   --mlir-print-debuginfo --mlir-print-local-scope %s | FileCheck %s

// Without a numerical profile the name stands alone.

// CHECK-LABEL: func.func @sdpa
//       CHECK:   iree_linalg_ext.attention
//       CHECK:   } -> tensor<2x3x5x6xf32> loc("composite torch.scaled_dot_product_attention"("model.py":20:7))
func.func @sdpa(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> loc("model.py":20:7)
  return %0 : tensor<2x3x5x6xf32>
}
func.func private @sdpa_impl(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>) -> tensor<2x3x5x7xf32>
  %1 = stablehlo.dot_general %0, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x3x5x7xf32>, tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32>
  return %1 : tensor<2x3x5x6xf32>
}
