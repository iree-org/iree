// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-custom-calls))" %s | FileCheck %s --check-prefix=DROP --implicit-check-not=shape_assertion --implicit-check-not=util.status.check_ok --implicit-check-not=stablehlo.reduce --implicit-check-not=scf.if
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx --compile-to=flow %s | FileCheck %s --check-prefix=FLOW --implicit-check-not=util.status.check_ok
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-device=metal --iree-metal-target-platform=macos --iree-metal-compile-to-metallib=false %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-device=vulkan --iree-vulkan-target=vp_android_baseline_2022 %s -o /dev/null

// Runtime assertions are dropped; the scatter keeps its original results.
// DROP-LABEL: func.func @scatter_effect
// DROP: %[[SCATTER:[a-zA-Z0-9_]+]] = "stablehlo.scatter"
// DROP: return %[[SCATTER]] : tensor<4xf32>
// FLOW-LABEL: util.func public @scatter_effect
// FLOW: util.return
func.func @scatter_effect(%operand: tensor<4xf32>, %indices: tensor<2x1xi32>, %updates: tensor<2xf32>) -> tensor<4xf32> {
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%current: tensor<f32>, %update: tensor<f32>):
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %pred = stablehlo.compare GE, %current, %zero : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%pred) {has_side_effect = true, error_message = "negative current"} : (tensor<i1>) -> ()
    stablehlo.return %update : tensor<f32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<4xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// Dropping assertions introduces no guards or status reduction around division.
// The asserted conditions (a positive current value and a nonzero update) are
// preconditions of the exported function: callers must meet them, so dividing
// unguarded is correct for every supported input.
// DROP-LABEL: func.func @scatter_divide
// DROP: stablehlo.divide
// DROP: return %{{[a-zA-Z0-9_]+}} : tensor<?xi32>
func.func @scatter_divide(%operand: tensor<?xi32>, %indices: tensor<?x1xi32>, %updates: tensor<?xi32>) -> tensor<?xi32> {
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%current: tensor<i32>, %update: tensor<i32>):
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %positive = stablehlo.compare GT, %current, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%positive) {has_side_effect = true, error_message = "nonpositive current"} : (tensor<i1>) -> ()
    %nonzero = stablehlo.compare NE, %update, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%nonzero) {has_side_effect = true, error_message = "zero update"} : (tensor<i1>) -> ()
    %quotient = stablehlo.divide %update, %current : tensor<i32>
    stablehlo.return %quotient : tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<?xi32>, tensor<?x1xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %result : tensor<?xi32>
}

// A mixed-type variadic scatter keeps both original results and their order.
// DROP-LABEL: func.func @scatter_variadic
// DROP: %[[PAIR:[a-zA-Z0-9_]+]]:2 = "stablehlo.scatter"
// DROP: return %[[PAIR]]#0, %[[PAIR]]#1 : tensor<4xf32>, tensor<4xi32>
func.func @scatter_variadic(%operand: tensor<4xf32>, %counts: tensor<4xi32>, %indices: tensor<2x1xi32>, %updates: tensor<2xf32>, %count_updates: tensor<2xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = "stablehlo.scatter"(%operand, %counts, %indices, %updates, %count_updates) ({
  ^bb0(%current: tensor<f32>, %count: tensor<i32>, %update: tensor<f32>, %count_update: tensor<i32>):
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %valid = stablehlo.compare GE, %count, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%valid) {has_side_effect = true, error_message = "negative count"} : (tensor<i1>) -> ()
    %next = stablehlo.add %current, %update : tensor<f32>
    %next_count = stablehlo.add %count, %count_update : tensor<i32>
    stablehlo.return %next, %next_count : tensor<f32>, tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<4xf32>, tensor<4xi32>, tensor<2x1xi32>, tensor<2xf32>, tensor<2xi32>) -> (tensor<4xf32>, tensor<4xi32>)
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}

// A scalar scatter keeps its scalar result without an extra status result.
// DROP-LABEL: func.func @scatter_scalar
// DROP: %[[SCALAR:[a-zA-Z0-9_]+]] = "stablehlo.scatter"
// DROP: return %[[SCALAR]] : tensor<f32>
func.func @scatter_scalar(%operand: tensor<f32>, %updates: tensor<1xf32>) -> tensor<f32> {
  %indices = stablehlo.constant dense<> : tensor<1x0xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%current: tensor<f32>, %update: tensor<f32>):
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %valid = stablehlo.compare GE, %current, %zero : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%valid) {has_side_effect = true, error_message = "negative scalar"} : (tensor<i1>) -> ()
    stablehlo.return %update : tensor<f32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [], scatter_dims_to_operand_dims = [], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<f32>, tensor<1x0xi32>, tensor<1xf32>) -> tensor<f32>
  return %result : tensor<f32>
}
