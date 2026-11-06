// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-stablehlo-to-linalg-ext))" %s | FileCheck %s --allow-empty

// A scatter the general lowering cannot take fails the pass instead of
// staying behind after its diagnostic.
// CHECK-NOT: stablehlo.scatter
func.func @scatter_unsigned_batched_indices(%input: tensor<2x5xi32>, %indices: tensor<2x3x1xui32>, %updates: tensor<2x3xi32>) -> tensor<2x5xi32> {
  // expected-error @+1 {{general scatter lowering requires unsigned index normalization}}
  %result = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%current: tensor<i32>, %update: tensor<i32>):
    stablehlo.return %update : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [1], input_batching_dims = [0],
      scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2>
  } : (tensor<2x5xi32>, tensor<2x3x1xui32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  return %result : tensor<2x5xi32>
}
