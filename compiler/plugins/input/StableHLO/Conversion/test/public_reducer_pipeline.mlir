// RUN: iree-compile --iree-input-type=stablehlo --compile-to=input %s | FileCheck %s

// CHECK-LABEL: util.func public @public_reducer_lowered
// CHECK: iree_linalg_ext.scan
// CHECK-LABEL: util.func public @public_reducer(
// CHECK: arith.addi
// CHECK: util.return
func.func @public_reducer_lowered(%input: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@public_reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func @public_reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}
