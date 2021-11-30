// RUN: iree-opt -iree-transformation-pipeline -iree-hal-target-backends=vmvx -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @add
func @add() -> i32 {
  %input = arith.constant 1 : i32
  %unf = util.do_not_optimize(%input) : i32
  // CHECK: vm.add.i32
  %result = arith.addi %unf, %unf : i32
  return %result : i32
}

// -----

// Ensure that add would normally be folded.
// CHECK-LABEL: @add_folded
func @add_folded() -> i32 {
  %input = arith.constant 1 : i32
  // CHECK-NOT: vm.add.i32
  %result = arith.addi %input, %input : i32
  return %result : i32
}

// -----

// CHECK-LABEL: @chained_add
func @chained_add() -> i32 {
  %input = arith.constant 1 : i32
  %unf = util.do_not_optimize(%input) : i32
  // CHECK: vm.add.i32
  %int = arith.addi %unf, %unf : i32
  // CHECK: vm.add.i32
  %result = arith.addi %int, %int : i32
  return %result : i32
}

// -----

// CHECK-LABEL: @unfoldable_constant
func @unfoldable_constant() -> i32 {
  %input = util.unfoldable_constant 1 : i32
  // CHECK: vm.add.i32
  %result = arith.addi %input, %input : i32
  return %result : i32
}
