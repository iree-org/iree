// RUN: iree-opt --split-input-file --verify-diagnostics --iree-stablehlo-preprocessing-flatten-cfg-tuples %s | FileCheck %s

// CHECK-LABEL: @flatten_func
module @flatten_func {
  // CHECK: func.func @caller(%arg0: i1, %arg1: tensor<f32>) -> tensor<f32>
  func.func @caller(%arg0 : i1, %arg1: tensor<f32>) -> tensor<f32> {
    // CHECK: %[[RESULT:.*]]:2 = call @callee(%arg0, %arg1, %arg1, %arg1) : (i1, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %0 = stablehlo.tuple %arg1, %arg1 : tuple<tensor<f32>, tensor<f32>>
    %1 = stablehlo.tuple %arg1, %0 : tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>
    %2 = call @callee(%arg0, %1) : (i1, tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>>
    %3 = stablehlo.get_tuple_element %2[0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
    // CHECK: return %[[RESULT]]#0 : tensor<f32>
    return %3 : tensor<f32>
  }

  // CHECK: func.func private @callee(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.func private @callee(%arg0: i1, %arg1: tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>> {
    // CHECK-NEXT: cf.cond_br %arg0, ^[[BB:.+]](%arg2, %arg3 : tensor<f32>, tensor<f32>), ^bb2
    // CHECK:      ^[[BB]](%[[V0:[^:]+]]: tensor<f32>, %[[V1:[^:]+]]: tensor<f32>)
    // CHECK-NEXT:   return %[[V0]], %[[V1]] : tensor<f32>, tensor<f32>
    %0 = stablehlo.get_tuple_element %arg1[0] : (tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tensor<f32>
    %1 = stablehlo.get_tuple_element %arg1[1] : (tuple<tensor<f32>, tuple<tensor<f32>, tensor<f32>>>) -> tuple<tensor<f32>, tensor<f32>>
    cf.cond_br %arg0, ^bb1(%1 : tuple<tensor<f32>, tensor<f32>>), ^bb2(%0 : tensor<f32>)
  ^bb1(%phi0 : tuple<tensor<f32>, tensor<f32>>):
    return %phi0 : tuple<tensor<f32>, tensor<f32>>
  ^bb2(%phi1 : tensor<f32>):
    %2 = stablehlo.tuple %phi1, %phi1 : tuple<tensor<f32>, tensor<f32>>
    cf.br ^bb1(%2 : tuple<tensor<f32>, tensor<f32>>)
  }
}

// -----

// Function recreation preserves metadata on the tensor following the tuple.
// CHECK-LABEL: func.func private @metadata_callee
// CHECK-SAME: tensor<3xf32>, %{{.+}}: tensor<3xf32>, %{{.+}}: tensor<3xf32> {iree.abi.output = 2 : index}
// CHECK-SAME: -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32> {iree.abi.tied = 2 : i64})
func.func private @metadata_callee(%t: tuple<tensor<3xf32>, tensor<3xf32>>, %x: tensor<3xf32> {iree.abi.output = 1 : index}) -> (tuple<tensor<3xf32>, tensor<3xf32>>, tensor<3xf32> {iree.abi.tied = 1 : i64}) {
  return %t, %x : tuple<tensor<3xf32>, tensor<3xf32>>, tensor<3xf32>
}
// CHECK-LABEL: func.func @metadata_caller
// CHECK: call @metadata_callee
// CHECK-SAME: <arg_attrs = [{}, {}, {tf.aliasing_output = 2 : i32}], res_attrs = [{}, {}, {iree.abi.tied = 2 : i64}], no_inline>
// CHECK-SAME: {iree.abi.tied_operands = [-1 : index, -1 : index, 2 : index], test.marker = "kept"}
func.func @metadata_caller(%t: tuple<tensor<3xf32>, tensor<3xf32>>, %x: tensor<3xf32>) -> tensor<3xf32> {
  %r:2 = "func.call"(%t, %x) <{callee = @metadata_callee, no_inline, arg_attrs = [{}, {tf.aliasing_output = 1 : i32}], res_attrs = [{}, {iree.abi.tied = 1 : i64}]}> {iree.abi.tied_operands = [-1 : index, 1 : index], test.marker = "kept"} : (tuple<tensor<3xf32>, tensor<3xf32>>, tensor<3xf32>) -> (tuple<tensor<3xf32>, tensor<3xf32>>, tensor<3xf32>)
  return %r#1 : tensor<3xf32>
}

// -----

// expected-error @+1 {{attributes on a tuple slot are unsupported}}
func.func private @tuple_slot_attributes(%t: tuple<tensor<3xf32>> {test.keep = 1 : i32}) -> tuple<tensor<3xf32>> {
  return %t : tuple<tensor<3xf32>>
}

// -----

func.func @indirect_tuple_call(%callee: (tuple<tensor<3xf32>>) -> tensor<3xf32>, %value: tuple<tensor<3xf32>>) -> tensor<3xf32> {
  // expected-error @+1 {{tuple signatures support only direct calls}}
  %r = func.call_indirect %callee(%value) : (tuple<tensor<3xf32>>) -> tensor<3xf32>
  return %r : tensor<3xf32>
}

// -----

func.func private @addressed_tuple(%t: tuple<tensor<3xf32>>) -> tuple<tensor<3xf32>> {
  return %t : tuple<tensor<3xf32>>
}
func.func @tuple_function_address() {
  // expected-error @+1 {{tuple signatures support only direct calls and result-dimension helper references}}
  %f = func.constant @addressed_tuple : (tuple<tensor<3xf32>>) -> tuple<tensor<3xf32>>
  return
}
