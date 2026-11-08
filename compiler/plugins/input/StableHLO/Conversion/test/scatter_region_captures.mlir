// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-stablehlo-to-linalg-ext))" %s | FileCheck %s
// RUN: iree-compile --split-input-file --iree-input-type=stablehlo --iree-hal-target-backends=vmvx --compile-to=input %s | FileCheck %s --check-prefix=INPUT

// Captures in nested control flow need the original tensor representation
// after the scatter's block arguments become scalars.
// CHECK-LABEL: func.func @guarded_window_scatter
// CHECK: iree_linalg_ext.scatter
// CHECK: tensor.from_elements
// CHECK: scf.if
// INPUT-LABEL: util.func public @guarded_window_scatter
// INPUT-NOT: unrealized_conversion_cast
func.func @guarded_window_scatter(%operand: tensor<4xf32>, %indices: tensor<2x1xi32>, %updates: tensor<2xf32>) -> tensor<4xf32> {
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%current: tensor<f32>, %update: tensor<f32>):
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %predicate = stablehlo.compare GE, %current, %zero : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %condition = tensor.extract %predicate[] : tensor<i1>
    %value = scf.if %condition -> tensor<f32> {
      scf.yield %update : tensor<f32>
    } else {
      scf.yield %current : tensor<f32>
    }
    stablehlo.return %value : tensor<f32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<4xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// The variadic fallback has the same capture/materialisation contract.
// CHECK-LABEL: func.func @guarded_general_scatter
// CHECK: iree_linalg_ext.general_scatter
// CHECK: tensor.from_elements
// CHECK: scf.if
// INPUT-LABEL: util.func public @guarded_general_scatter
// INPUT-NOT: unrealized_conversion_cast
func.func @guarded_general_scatter(%operand: tensor<4xf32>, %state: tensor<4xi32>, %indices: tensor<2x1xi32>, %updates: tensor<2xf32>, %state_updates: tensor<2xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = "stablehlo.scatter"(%operand, %state, %indices, %updates, %state_updates) ({
  ^bb0(%current: tensor<f32>, %status: tensor<i32>, %update: tensor<f32>, %next_status: tensor<i32>):
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %predicate = stablehlo.compare EQ, %status, %zero : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %condition = tensor.extract %predicate[] : tensor<i1>
    %next:2 = scf.if %condition -> (tensor<f32>, tensor<i32>) {
      scf.yield %update, %next_status : tensor<f32>, tensor<i32>
    } else {
      scf.yield %current, %status : tensor<f32>, tensor<i32>
    }
    stablehlo.return %next#0, %next#1 : tensor<f32>, tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<4xf32>, tensor<4xi32>, tensor<2x1xi32>, tensor<2xf32>, tensor<2xi32>) -> (tensor<4xf32>, tensor<4xi32>)
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}
