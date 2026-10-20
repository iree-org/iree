// RUN: not iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx --compile-to=flow %s -o /dev/null 2>&1 | FileCheck %s

// Partial-window scatter lowering cannot retain this effectful update region.
// Reject the unsupported form instead of silently discarding its assertion.
// CHECK: error: 'iree_linalg_ext.scatter' op mismatch in shape of indices and update value at dim#1
func.func @effectful_partial_scatter(%operand: tensor<4xf32>, %updates: tensor<2xf32>) -> tensor<4xf32> {
  %indices = stablehlo.constant dense<1> : tensor<1xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%current: tensor<f32>, %update: tensor<f32>):
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %pred = stablehlo.compare GE, %current, %zero : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%pred) {has_side_effect = true, error_message = "negative current"} : (tensor<i1>) -> ()
    stablehlo.return %update : tensor<f32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 0>, indices_are_sorted = false, unique_indices = false} : (tensor<4xf32>, tensor<1xi32>, tensor<2xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
