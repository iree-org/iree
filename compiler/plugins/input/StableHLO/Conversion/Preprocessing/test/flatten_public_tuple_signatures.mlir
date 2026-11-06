// RUN: iree-opt --split-input-file --verify-diagnostics \
// RUN:   --iree-stablehlo-preprocessing-flatten-public-tuple-signatures %s | FileCheck %s

// Public tuple slots become their leaves in depth-first order; empty tuples
// vanish. The declaration keeps the source signature, tokens shown as fences.
// CHECK-LABEL: func.func @nested
// CHECK-SAME:    (%[[A:[^:]+]]: tensor<2x3xf32>, %[[T:[^:]+]]: !stablehlo.token, %[[B:[^:]+]]: tensor<5x4xi32>)
// CHECK-SAME:    -> (tensor<5x4xi32>, tensor<2x3xf32>)
// CHECK-SAME:    iree.abi.declaration = "sync func @nested(%input0: tuple<tensor<2x3xf32>, tuple<!hal.fence, tuple<>>>, %input1: tensor<5x4xi32>) -> (%output0: tuple<tensor<5x4xi32>, tensor<2x3xf32>>)"
// CHECK:         %[[EMPTY:.+]] = stablehlo.tuple  : tuple<>
// CHECK:         %[[INNER:.+]] = stablehlo.tuple %[[T]], %[[EMPTY]]
// CHECK:         %[[OUTER:.+]] = stablehlo.tuple %[[A]], %[[INNER]]
// CHECK:         %[[X:.+]] = stablehlo.get_tuple_element %[[OUTER]][0]
// CHECK:         %[[RESULT:.+]] = stablehlo.tuple %[[B]], %[[X]]
// CHECK:         %[[R0:.+]] = stablehlo.get_tuple_element %[[RESULT]][0]
// CHECK:         %[[R1:.+]] = stablehlo.get_tuple_element %[[RESULT]][1]
// CHECK:         return %[[R0]], %[[R1]]
func.func @nested(%arg0: tuple<tensor<2x3xf32>, tuple<!stablehlo.token, tuple<>>>, %arg1: tensor<5x4xi32>) -> tuple<tensor<5x4xi32>, tensor<2x3xf32>> {
  %x = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<2x3xf32>, tuple<!stablehlo.token, tuple<>>>) -> tensor<2x3xf32>
  %r = stablehlo.tuple %arg1, %x : tuple<tensor<5x4xi32>, tensor<2x3xf32>>
  return %r : tuple<tensor<5x4xi32>, tensor<2x3xf32>>
}

// Internal calls pass leaves and rebuild the result tuple; private functions
// keep their tuple signatures.
// CHECK-LABEL: func.func private @caller
// CHECK-SAME:    (%{{.+}}: tuple<tensor<2x3xf32>, tuple<!stablehlo.token, tuple<>>>, %{{.+}}: tensor<5x4xi32>) -> tuple<tensor<5x4xi32>, tensor<2x3xf32>>
// CHECK:         %[[CALL:.+]]:2 = call @nested(%{{.+}}, %{{.+}}, %{{.+}}) : (tensor<2x3xf32>, !stablehlo.token, tensor<5x4xi32>) -> (tensor<5x4xi32>, tensor<2x3xf32>)
// CHECK:         stablehlo.tuple %[[CALL]]#0, %[[CALL]]#1
func.func private @caller(%arg0: tuple<tensor<2x3xf32>, tuple<!stablehlo.token, tuple<>>>, %arg1: tensor<5x4xi32>) -> tuple<tensor<5x4xi32>, tensor<2x3xf32>> {
  %r = call @nested(%arg0, %arg1) : (tuple<tensor<2x3xf32>, tuple<!stablehlo.token, tuple<>>>, tensor<5x4xi32>) -> tuple<tensor<5x4xi32>, tensor<2x3xf32>>
  return %r : tuple<tensor<5x4xi32>, tensor<2x3xf32>>
}

// -----

// Slot attributes follow their leaf, and alias indices are remapped to the
// flattened results.
// CHECK-LABEL: func.func @aliasing
// CHECK-SAME:    %{{.+}}: tensor<2x3xf32>, %{{.+}}: tensor<4x5xf32>, %{{.+}}: tensor<3x7xi32> {tf.aliasing_output = 2 : i32}
func.func @aliasing(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xf32>>, %arg1: tensor<3x7xi32> {tf.aliasing_output = 1 : i32}) -> (tuple<tensor<2x3xf32>, tensor<4x5xf32>>, tensor<3x7xi32>) {
  return %arg0, %arg1 : tuple<tensor<2x3xf32>, tensor<4x5xf32>>, tensor<3x7xi32>
}

// -----

// expected-error @+1 {{attributes on a tuple slot are unsupported}}
func.func @tuple_slot_attrs(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xf32>> {tf.aliasing_output = 0 : i32}) -> tuple<tensor<2x3xf32>, tensor<4x5xf32>> {
  return %arg0 : tuple<tensor<2x3xf32>, tensor<4x5xf32>>
}

// -----

// expected-error @+1 {{'tf.aliasing_output' cannot refer to a tuple slot}}
func.func @alias_into_tuple(%arg0: tensor<2x3xf32> {tf.aliasing_output = 0 : i32}) -> tuple<tensor<2x3xf32>, tensor<4x5xf32>> {
  %e = stablehlo.constant dense<0.0> : tensor<4x5xf32>
  %r = stablehlo.tuple %arg0, %e : tuple<tensor<2x3xf32>, tensor<4x5xf32>>
  return %r : tuple<tensor<2x3xf32>, tensor<4x5xf32>>
}

// -----

// Callers keep no_inline and discardable attributes; slot metadata after a
// tuple moves to its leaf, with alias indices and ties remapped.
// CHECK-LABEL: func.func private @metadata_caller
// CHECK:         call @metadata_callee(%{{[^)]+}})
// CHECK-SAME:      <arg_attrs = [{}, {}, {test.slot = "arg", tf.aliasing_output = 2 : i32}], res_attrs = [{}, {}, {iree.abi.tied = 2 : i32}], no_inline>
// CHECK-SAME:      {iree.abi.tied_operands = [-1 : index, -1 : index, 2 : index], test.marker = "kept"}
func.func @metadata_callee(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xf32>>, %arg1: tensor<3x7xi32>) -> (tuple<tensor<2x3xf32>, tensor<4x5xf32>>, tensor<3x7xi32>) {
  return %arg0, %arg1 : tuple<tensor<2x3xf32>, tensor<4x5xf32>>, tensor<3x7xi32>
}
func.func private @metadata_caller(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xf32>>, %arg1: tensor<3x7xi32>) -> tensor<3x7xi32> {
  %r:2 = "func.call"(%arg0, %arg1) <{callee = @metadata_callee, no_inline, arg_attrs = [{}, {tf.aliasing_output = 1 : i32, test.slot = "arg"}], res_attrs = [{}, {iree.abi.tied = 1 : i32}]}> {iree.abi.tied_operands = [-1 : index, 1 : index], test.marker = "kept"} : (tuple<tensor<2x3xf32>, tensor<4x5xf32>>, tensor<3x7xi32>) -> (tuple<tensor<2x3xf32>, tensor<4x5xf32>>, tensor<3x7xi32>)
  return %r#1 : tensor<3x7xi32>
}

// -----

func.func @call_slot_callee(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xf32>>) -> tensor<4x5xf32> {
  %x = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<2x3xf32>, tensor<4x5xf32>>) -> tensor<4x5xf32>
  return %x : tensor<4x5xf32>
}
func.func private @call_slot_caller(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xf32>>) -> tensor<4x5xf32> {
  // expected-error @+1 {{attributes on a tuple slot are unsupported}}
  %r = "func.call"(%arg0) <{callee = @call_slot_callee, arg_attrs = [{test.slot = "tuple"}]}> : (tuple<tensor<2x3xf32>, tensor<4x5xf32>>) -> tensor<4x5xf32>
  return %r : tensor<4x5xf32>
}

// -----

func.func @tied_tuple_callee(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xf32>>) -> tuple<tensor<2x3xf32>, tensor<4x5xf32>> {
  return %arg0 : tuple<tensor<2x3xf32>, tensor<4x5xf32>>
}
func.func private @tied_tuple_caller(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xf32>>) -> tuple<tensor<2x3xf32>, tensor<4x5xf32>> {
  // expected-error @+1 {{'iree.abi.tied_operands' cannot refer to an expanded signature slot}}
  %r = call @tied_tuple_callee(%arg0) {iree.abi.tied_operands = [0 : index]} : (tuple<tensor<2x3xf32>, tensor<4x5xf32>>) -> tuple<tensor<2x3xf32>, tensor<4x5xf32>>
  return %r : tuple<tensor<2x3xf32>, tensor<4x5xf32>>
}

// -----

// The declaration names arguments by iree.abi.name and keeps their IO
// attributes, as the native ABI wrapper prints them.
// CHECK-LABEL: func.func @named
// CHECK-SAME:    iree.abi.declaration = "sync func @named(%input0: tuple<tensor<2x3xf32>, tensor<4x5xi8>>, %weights: tensor<5x4xi32> {test.io = 1 : i32}) -> (%output0: tuple<tensor<5x4xi32>>)"
func.func @named(%arg0: tuple<tensor<2x3xf32>, tensor<4x5xi8>>, %arg1: tensor<5x4xi32> {iree.abi.name = "weights", test.io = 1 : i32}) -> tuple<tensor<5x4xi32>> {
  %r = stablehlo.tuple %arg1 : tuple<tensor<5x4xi32>>
  return %r : tuple<tensor<5x4xi32>>
}

// -----

// A symbol in a call's unrelated metadata does not make it a call to the
// public tuple function.
func.func @referenced_tuple(%t: tuple<tensor<3xf32>>) -> tensor<3xf32> {
  %x = stablehlo.get_tuple_element %t[0] : (tuple<tensor<3xf32>>) -> tensor<3xf32>
  return %x : tensor<3xf32>
}
func.func private @other(%x: tensor<3xf32>) -> tensor<3xf32>
func.func private @symbol_reference(%x: tensor<3xf32>) -> tensor<3xf32> {
  // expected-error @+1 {{public tuple signatures support only direct calls}}
  %r = func.call @other(%x) {test.reference = @referenced_tuple} : (tensor<3xf32>) -> tensor<3xf32>
  return %r : tensor<3xf32>
}

// -----

// Two symbol uses on one call must not schedule that call for erasure twice.
func.func @duplicated_use(%t: tuple<tensor<3xf32>, tensor<3xf32>>) -> tuple<tensor<3xf32>, tensor<3xf32>> {
  return %t : tuple<tensor<3xf32>, tensor<3xf32>>
}
// CHECK-LABEL: func.func private @duplicated_use_caller
// CHECK: %[[CALL:.+]]:2 = call @duplicated_use
// CHECK-SAME: {test.reference = @duplicated_use}
// CHECK-NOT: call @duplicated_use
// CHECK: return
func.func private @duplicated_use_caller(%t: tuple<tensor<3xf32>, tensor<3xf32>>) -> tuple<tensor<3xf32>, tensor<3xf32>> {
  %r = func.call @duplicated_use(%t) {test.reference = @duplicated_use} : (tuple<tensor<3xf32>, tensor<3xf32>>) -> tuple<tensor<3xf32>, tensor<3xf32>>
  return %r : tuple<tensor<3xf32>, tensor<3xf32>>
}
