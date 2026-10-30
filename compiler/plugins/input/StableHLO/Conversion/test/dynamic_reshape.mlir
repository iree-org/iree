// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s

// CHECK-LABEL: @collapse_fusion
// CHECK: tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2]]
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
