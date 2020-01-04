// RUN: iree-opt -iree-transformation-pipeline -split-input-file %s | IreeFileCheck %s

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
