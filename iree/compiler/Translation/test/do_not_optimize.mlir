// RUN: iree-opt -iree-transformation-pipeline -iree-hal-target-backends=vmla -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @add
func @add() -> i32 {
  %input = constant 1 : i32
  %unf = iree.do_not_optimize(%input) : i32
  // CHECK: vm.add.i32
  %result = addi %unf, %unf : i32
  return %result : i32
}

// -----

// Ensure that add would normally be folded.
// CHECK-LABEL: @add_folded
func @add_folded() -> i32 {
  %input = constant 1 : i32
  // CHECK-NOT: vm.add.i32
  %result = addi %input, %input : i32
  return %result : i32
}

// -----

// CHECK-LABEL: @chained_add
func @chained_add() -> i32 {
  %input = constant 1 : i32
  %unf = iree.do_not_optimize(%input) : i32
  // CHECK: vm.add.i32
  %int = addi %unf, %unf : i32
  // CHECK: vm.add.i32
  %result = addi %int, %int : i32
  return %result : i32
}

// -----

// CHECK-LABEL: @unfoldable_constant
func @unfoldable_constant() -> i32 {
  %input = iree.unfoldable_constant 1 : i32
  // CHECK: vm.add.i32
  %result = addi %input, %input : i32
  return %result : i32
}

// -----

// CHECK-LABEL: vm.rodata @dynamic_constant_const_0 dense<3.000000e+00> : tensor<2x3xf32>
// CHECK: vm.func @dynamic_constant
func @dynamic_constant() -> tensor<?x?xf32> {
  // CHECK: vm.call @hal.buffer_view.dim
  %input = iree.dynamic_shape_constant dense<3.0> : tensor<2x3xf32> -> tensor<?x?xf32>
  %res = "mhlo.abs"(%input) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %res : tensor<?x?xf32>
}
