// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s --implicit-check-not=stablehlo.
// RUN: iree-compile --split-input-file --iree-hal-target-backends=llvm-cpu --compile-to=flow %s | FileCheck %s --check-prefix=FLOW

// CHECK-LABEL: @collapse_fusion
// CHECK: tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2]]
// FLOW-LABEL: util.func public @collapse_fusion
// FLOW: flow.dispatch @
// FLOW-NOT: flow.dispatch @
// FLOW: util.return
func.func @collapse_fusion(%a: tensor<?x?x64xf32>) -> tensor<?x64xf32> {
  %b = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x64xf32>) -> tensor<i32>
  %s = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x64xf32>) -> tensor<i32>
  %bs = stablehlo.multiply %b, %s : tensor<i32>
  %bs_vec = stablehlo.reshape %bs : (tensor<i32>) -> tensor<1xi32>
  %c64 = stablehlo.constant dense<64> : tensor<1xi32>
  %shape = stablehlo.concatenate %bs_vec, %c64, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %exp = stablehlo.exponential %a : tensor<?x?x64xf32>
  %view = stablehlo.dynamic_reshape %exp, %shape : (tensor<?x?x64xf32>, tensor<2xi32>) -> tensor<?x64xf32>
  %square = stablehlo.multiply %view, %view : tensor<?x64xf32>
  return %square : tensor<?x64xf32>
}

// -----

// CHECK-LABEL: @expand
// CHECK: tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2]] output_shape [{{.*}}] : tensor<?x64xf32> into tensor<?x?x64xf32>
func.func @expand(%a: tensor<?x64xf32>, %shape: tensor<3xi64>) -> tensor<?x?x64xf32> {
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x64xf32>, tensor<3xi64>) -> tensor<?x?x64xf32>
  return %r : tensor<?x?x64xf32>
}

// -----

// CHECK-LABEL: @insert_unit_dim
// CHECK: tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2]] output_shape [1, %{{.+}}, 64]
func.func @insert_unit_dim(%a: tensor<?x64xf32>, %shape: tensor<3xi32>) -> tensor<1x?x64xf32> {
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x64xf32>, tensor<3xi32>) -> tensor<1x?x64xf32>
  return %r : tensor<1x?x64xf32>
}

// -----

// Element count alone cannot determine both result dimensions.
// CHECK-LABEL: @ambiguous_collapse
// CHECK: tensor.reshape
func.func @ambiguous_collapse(%a: tensor<?x?x?xf32>, %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// A zero dimension leaves the other dimensions unconstrained.
// CHECK-LABEL: @zero_collapse
// CHECK: tensor.reshape
func.func @zero_collapse(%a: tensor<?x?x0xf32>, %shape: tensor<2xi64>) -> tensor<?x0xf32> {
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?x0xf32>, tensor<2xi64>) -> tensor<?x0xf32>
  return %r : tensor<?x0xf32>
}

// -----

// CHECK-LABEL: @zero_expand
// CHECK: tensor.reshape
func.func @zero_expand(%a: tensor<?x0xf32>, %shape: tensor<3xi64>) -> tensor<?x?x0xf32> {
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x0xf32>, tensor<3xi64>) -> tensor<?x?x0xf32>
  return %r : tensor<?x?x0xf32>
}

// -----

// Shape products of the operand's own sizes group contiguously. JAX checks each
// dimension variable, which ties the operand to the i32 sizes it reads.
// CHECK-LABEL: @collapse_products
// CHECK: tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
// CHECK-NOT: tensor.reshape
// FLOW-LABEL: util.func public @collapse_products
// FLOW: flow.dispatch @
// FLOW-NOT: flow.dispatch @
// FLOW: util.return
func.func @collapse_products(%a: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %size0 = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %ok0 = stablehlo.compare GE, %size0, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok0) {error_message = "dim 0 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %size1 = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %ok1 = stablehlo.compare GE, %size1, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok1) {error_message = "dim 1 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %size2 = stablehlo.get_dimension_size %a, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %ok2 = stablehlo.compare GE, %size2, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok2) {error_message = "dim 2 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %b = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %s = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %d = stablehlo.get_dimension_size %a, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %bs = stablehlo.multiply %b, %s : tensor<i32>
  %bs_vec = stablehlo.reshape %bs : (tensor<i32>) -> tensor<1xi32>
  %d_vec = stablehlo.reshape %d : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %bs_vec, %d_vec, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %exp = stablehlo.exponential %a : tensor<?x?x?xf32>
  %view = stablehlo.dynamic_reshape %exp, %shape : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %square = stablehlo.multiply %view, %view : tensor<?x?xf32>
  return %square : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @collapse_leading_unit
// CHECK: tensor.collapse_shape %{{.+}} {{\[}}[0, 1, 2], [3]] : tensor<1x?x?x?xf32> into tensor<?x?xf32>
func.func @collapse_leading_unit(%a: tensor<1x?x?x?xf32>) -> tensor<?x?xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %size1 = stablehlo.get_dimension_size %a, dim = 1 : (tensor<1x?x?x?xf32>) -> tensor<i32>
  %ok1 = stablehlo.compare GE, %size1, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok1) {error_message = "dim 1 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %size2 = stablehlo.get_dimension_size %a, dim = 2 : (tensor<1x?x?x?xf32>) -> tensor<i32>
  %ok2 = stablehlo.compare GE, %size2, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok2) {error_message = "dim 2 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %size3 = stablehlo.get_dimension_size %a, dim = 3 : (tensor<1x?x?x?xf32>) -> tensor<i32>
  %ok3 = stablehlo.compare GE, %size3, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok3) {error_message = "dim 3 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %b = stablehlo.get_dimension_size %a, dim = 1 : (tensor<1x?x?x?xf32>) -> tensor<i32>
  %s = stablehlo.get_dimension_size %a, dim = 2 : (tensor<1x?x?x?xf32>) -> tensor<i32>
  %d = stablehlo.get_dimension_size %a, dim = 3 : (tensor<1x?x?x?xf32>) -> tensor<i32>
  %bs = stablehlo.multiply %s, %b : tensor<i32>
  %bs_vec = stablehlo.reshape %bs : (tensor<i32>) -> tensor<1xi32>
  %d_vec = stablehlo.reshape %d : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %bs_vec, %d_vec, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<1x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Two dynamic groups, one with a constant factor.
// CHECK-LABEL: @collapse_two_groups
// CHECK: tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2, 3]] : tensor<?x?x4x?xf32> into tensor<?x?xf32>
func.func @collapse_two_groups(%a: tensor<?x?x4x?xf32>) -> tensor<?x?xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %size0 = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x4x?xf32>) -> tensor<i32>
  %ok0 = stablehlo.compare GE, %size0, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok0) {error_message = "dim 0 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %size1 = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x4x?xf32>) -> tensor<i32>
  %ok1 = stablehlo.compare GE, %size1, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok1) {error_message = "dim 1 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %size3 = stablehlo.get_dimension_size %a, dim = 3 : (tensor<?x?x4x?xf32>) -> tensor<i32>
  %ok3 = stablehlo.compare GE, %size3, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok3) {error_message = "dim 3 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %d0 = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x4x?xf32>) -> tensor<i32>
  %d1 = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x4x?xf32>) -> tensor<i32>
  %d3 = stablehlo.get_dimension_size %a, dim = 3 : (tensor<?x?x4x?xf32>) -> tensor<i32>
  %p0 = stablehlo.multiply %d0, %d1 : tensor<i32>
  %p1 = stablehlo.multiply %c4, %d3 : tensor<i32>
  %v0 = stablehlo.reshape %p0 : (tensor<i32>) -> tensor<1xi32>
  %v1 = stablehlo.reshape %p1 : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %v0, %v1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?x4x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Flattening and restoring the same sizes is a collapse and an expand.
// CHECK-LABEL: @flatten_round_trip
// CHECK: tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
// CHECK: tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2]] output_shape [{{.*}}] : tensor<?x?xf32> into tensor<?x?x?xf32>
// CHECK-NOT: tensor.reshape
// FLOW-LABEL: util.func public @flatten_round_trip
// FLOW: flow.dispatch @
// FLOW-NOT: flow.dispatch @
// FLOW: util.return
func.func @flatten_round_trip(%a: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %size0 = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %ok0 = stablehlo.compare GE, %size0, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok0) {error_message = "dim 0 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %size1 = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %ok1 = stablehlo.compare GE, %size1, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok1) {error_message = "dim 1 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %size2 = stablehlo.get_dimension_size %a, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %ok2 = stablehlo.compare GE, %size2, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok2) {error_message = "dim 2 >= 1", has_side_effect = true} : (tensor<i1>) -> ()
  %b = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %s = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %d = stablehlo.get_dimension_size %a, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %bs = stablehlo.multiply %b, %s : tensor<i32>
  %bs_vec = stablehlo.reshape %bs : (tensor<i32>) -> tensor<1xi32>
  %b_vec = stablehlo.reshape %b : (tensor<i32>) -> tensor<1xi32>
  %s_vec = stablehlo.reshape %s : (tensor<i32>) -> tensor<1xi32>
  %d_vec = stablehlo.reshape %d : (tensor<i32>) -> tensor<1xi32>
  %flat_shape = stablehlo.concatenate %bs_vec, %d_vec, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %shape = stablehlo.concatenate %b_vec, %s_vec, %d_vec, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %flat = stablehlo.dynamic_reshape %a, %flat_shape : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %exp = stablehlo.exponential %flat : tensor<?x?xf32>
  %r = stablehlo.dynamic_reshape %exp, %shape : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// A product across groups in the wrong order proves no contiguous grouping.
// CHECK-LABEL: @collapse_out_of_order
// CHECK: tensor.reshape
func.func @collapse_out_of_order(%a: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %d0 = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %d1 = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %d2 = stablehlo.get_dimension_size %a, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %p = stablehlo.multiply %d0, %d2 : tensor<i32>
  %v0 = stablehlo.reshape %d1 : (tensor<i32>) -> tensor<1xi32>
  %v1 = stablehlo.reshape %p : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %v0, %v1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @collapse_constant_mismatch
// CHECK: tensor.reshape
func.func @collapse_constant_mismatch(%a: tensor<?x6x?xf32>) -> tensor<?x?xf32> {
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %d0 = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x6x?xf32>) -> tensor<i32>
  %d2 = stablehlo.get_dimension_size %a, dim = 2 : (tensor<?x6x?xf32>) -> tensor<i32>
  %p = stablehlo.multiply %d0, %c4 : tensor<i32>
  %v0 = stablehlo.reshape %p : (tensor<i32>) -> tensor<1xi32>
  %v1 = stablehlo.reshape %d2 : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %v0, %v1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x6x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @expand_division
// CHECK: tensor.reshape
func.func @expand_division(%a: tensor<?x?xf32>, %n: tensor<i32>) -> tensor<?x?x?xf32> {
  %bs = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %d = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %b = stablehlo.divide %bs, %n : tensor<i32>
  %b1 = stablehlo.reshape %b : (tensor<i32>) -> tensor<1xi32>
  %n1 = stablehlo.reshape %n : (tensor<i32>) -> tensor<1xi32>
  %d1 = stablehlo.reshape %d : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %b1, %n1, %d1, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// Without a check, an i32 size is a truncation of the dimension, which proves
// nothing about the dimension itself.
// CHECK-LABEL: @collapse_unchecked
// CHECK: tensor.reshape
func.func @collapse_unchecked(%a: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %b = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %s = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %d = stablehlo.get_dimension_size %a, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %bs = stablehlo.multiply %b, %s : tensor<i32>
  %bs_vec = stablehlo.reshape %bs : (tensor<i32>) -> tensor<1xi32>
  %d_vec = stablehlo.reshape %d : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %bs_vec, %d_vec, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Equal sizes alone allow an empty tensor: for [65536, 65536, 0] the i32
// product wraps to 0, a valid target the collapse would change to 2^32.
// CHECK-LABEL: @collapse_wrapping_product_of_possibly_empty
// CHECK: tensor.reshape
func.func @collapse_wrapping_product_of_possibly_empty(%a: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %a0 = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %a1 = stablehlo.get_dimension_size %a, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %a2 = stablehlo.get_dimension_size %a, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %b0 = stablehlo.get_dimension_size %b, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %b1 = stablehlo.get_dimension_size %b, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %b2 = stablehlo.get_dimension_size %b, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %eq0 = stablehlo.compare EQ, %a0, %b0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%eq0) {error_message = "dim 0", has_side_effect = true} : (tensor<i1>) -> ()
  %eq1 = stablehlo.compare EQ, %a1, %b1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%eq1) {error_message = "dim 1", has_side_effect = true} : (tensor<i1>) -> ()
  %eq2 = stablehlo.compare EQ, %a2, %b2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%eq2) {error_message = "dim 2", has_side_effect = true} : (tensor<i1>) -> ()
  %p = stablehlo.multiply %a0, %a1 : tensor<i32>
  %v0 = stablehlo.reshape %p : (tensor<i32>) -> tensor<1xi32>
  %v1 = stablehlo.reshape %a2 : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %v0, %v1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %r = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Shape expressions may use wider integer intermediates before narrowing to
// i64. A constant outside the proof's signed 64-bit domain must leave the generic
// reshape intact, including when compiler assertions are enabled.
// CHECK-LABEL: @negative_collapse_wide_constant
// CHECK: tensor.reshape
// CHECK-NOT: tensor.collapse_shape
// FLOW-LABEL: util.func public @negative_collapse_wide_constant
func.func @negative_collapse_wide_constant(%a: tensor<?x?x?xf32>, %wide: i128) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %large = arith.constant 1267650600228229401496703205377 : i128
  %sum = arith.addi %wide, %large : i128
  %size = arith.trunci %sum : i128 to i64
  %d = tensor.dim %a, %c2 : tensor<?x?x?xf32>
  %di = arith.index_cast %d : index to i64
  %shape = tensor.from_elements %size, %di : tensor<2xi64>
  %result = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// -----

// A wide-typed constant that fits in the proof domain still permits grouping.
// CHECK-LABEL: @collapse_wide_typed_factor
// CHECK: tensor.collapse_shape %{{.+}} {{\[}}[0, 1, 2], [3]]
// CHECK-NOT: tensor.reshape
// FLOW-LABEL: util.func public @collapse_wide_typed_factor
func.func @collapse_wide_typed_factor(%a: tensor<2x?x?x?xf32>) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %two = arith.constant 2 : i128
  %d1 = tensor.dim %a, %c1 : tensor<2x?x?x?xf32>
  %d2 = tensor.dim %a, %c2 : tensor<2x?x?x?xf32>
  %d3 = tensor.dim %a, %c3 : tensor<2x?x?x?xf32>
  %product = arith.muli %d1, %d2 : index
  %wide = arith.index_cast %product : index to i128
  %scaled = arith.muli %wide, %two : i128
  %size = arith.trunci %scaled : i128 to i64
  %last = arith.index_cast %d3 : index to i64
  %shape = tensor.from_elements %size, %last : tensor<2xi64>
  %result = stablehlo.dynamic_reshape %a, %shape : (tensor<2x?x?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// -----

// An out-of-bounds tensor.extract has undefined behaviour when executed, but is
// verifier-valid. The optional proof must leave it and the generic reshape
// intact instead of indexing the tensor.from_elements operands out of bounds.
// CHECK-LABEL: @negative_collapse_extract_index
// CHECK: tensor.extract
// CHECK: tensor.reshape
// CHECK-NOT: tensor.collapse_shape
// FLOW-LABEL: util.func public @negative_collapse_extract_index
func.func @negative_collapse_extract_index(%a: tensor<?x?x?xf32>, %s0: i64, %s1: i64) -> tensor<?x?xf32> {
  %negative = arith.constant -1 : index
  %sizes = tensor.from_elements %s0, %s1 : tensor<2xi64>
  %bad = tensor.extract %sizes[%negative] : tensor<2xi64>
  %shape = tensor.from_elements %bad, %s1 : tensor<2xi64>
  %result = stablehlo.dynamic_reshape %a, %shape : (tensor<?x?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
