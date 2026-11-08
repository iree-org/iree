// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-custom-calls))" %s | FileCheck %s --check-prefix=STATUS
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx --compile-to=flow %s | FileCheck %s --check-prefix=FLOW
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-device=metal --iree-metal-target-platform=macos --iree-metal-compile-to-metallib=false %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-device=vulkan --iree-vulkan-target=vp_android_baseline_2022 %s -o /dev/null

// A coupled status result survives repeated indices. The host observes failure
// after the dispatches, so no shader status-check operation is required.
// STATUS-LABEL: func.func @scatter_effect
// STATUS: %[[SCATTER:.*]]:2 = "stablehlo.scatter"
// STATUS: scf.if
// STATUS: scf.if
// STATUS-NOT: util.status.check_ok
// STATUS-NOT: stablehlo.custom_call
// STATUS: stablehlo.reduce(%[[SCATTER]]#1
// STATUS: util.status.check_ok {{.*}}"negative current"
// STATUS: return %[[SCATTER]]#0
// FLOW-NOT: util.status.check_ok
// FLOW-LABEL: util.func public @scatter_effect
// FLOW: flow.tensor.load
// FLOW: util.status.check_ok {{.*}}"negative current"
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

// The division stays inside both success continuations. Distinct failures retain
// their original messages; a zero-length status tensor reduces to success.
// STATUS-LABEL: func.func @scatter_divide
// STATUS: %[[DIV_SCATTER:.*]]:2 = "stablehlo.scatter"
// STATUS: scf.if
// STATUS: scf.if
// STATUS: scf.if
// STATUS: %[[QUOTIENT:.*]] = stablehlo.divide
// STATUS-NEXT: scf.yield %[[QUOTIENT]],
// STATUS-NOT: util.status.check_ok
// STATUS: stablehlo.reduce(%[[DIV_SCATTER]]#1
// STATUS: util.status.check_ok {{.*}}"nonpositive current"
// STATUS: util.status.check_ok {{.*}}"zero update"
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

// The status component follows all originals and updates of a mixed-type
// variadic scatter without changing their ordering or result types.
// STATUS-LABEL: func.func @scatter_variadic
// STATUS: %[[PAIR:.*]]:3 = "stablehlo.scatter"
// STATUS: -> (tensor<4xf32>, tensor<4xi32>, tensor<4xi32>)
// STATUS: stablehlo.reduce(%[[PAIR]]#2
// STATUS: util.status.check_ok {{.*}}"negative count"
// STATUS: return %[[PAIR]]#0, %[[PAIR]]#1
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

// A scalar original has a scalar status result and an empty reduction axis set.
// STATUS-LABEL: func.func @scatter_scalar
// STATUS: %[[SCALAR:.*]]:2 = "stablehlo.scatter"
// STATUS: -> (tensor<f32>, tensor<i32>)
// STATUS: util.status.check_ok {{.*}}"negative scalar"
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
