// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx --compile-to=flow %s | FileCheck %s
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu %s -o /dev/null

// The assertion inside the scatter region must survive dispatch formation.
// CHECK: flow.executable
// CHECK: util.status.check_ok {{.*}}"negative current"
func.func @scatter_effect(%operand: tensor<4xf32>, %updates: tensor<2xf32>) -> tensor<4xf32> {
  %indices = stablehlo.constant dense<[[1], [2]]> : tensor<2x1xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%current: tensor<f32>, %update: tensor<f32>):
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %pred = stablehlo.compare GE, %current, %zero : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%pred) {has_side_effect = true, error_message = "negative current"} : (tensor<i1>) -> ()
    stablehlo.return %update : tensor<f32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<4xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
