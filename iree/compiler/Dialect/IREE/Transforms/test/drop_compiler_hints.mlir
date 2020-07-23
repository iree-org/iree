// RUN: iree-opt -split-input-file -iree-drop-compiler-hints %s | IreeFileCheck --implicit-check-not="iree.do_not_optimize" %s

// This file is used as an example in docs/developing_iree/developer_overview.md.
// If you move or delete it, please update the documentation accordingly.

// CHECK-LABEL: @constant
func @constant() -> i32 {
  // CHECK-NEXT: %[[C1:.+]] = constant 1
  %c1 = constant 1 : i32
  %0 = iree.do_not_optimize(%c1) : i32
  // CHECK-NEXT: return %[[C1]]
  return %0 : i32
}

// -----

// CHECK-LABEL: @multiple
func @multiple() -> (i32, i32) {
  // CHECK-NEXT: %[[C1:.+]] = constant 1
  %c1 = constant 1 : i32
  %0 = iree.do_not_optimize(%c1) : i32
  %1 = iree.do_not_optimize(%0) : i32
  // CHECK-NEXT: %[[C2:.+]] = constant 2
  %c2 = constant 2 : i32
  %2 = iree.do_not_optimize(%1) : i32
  %3 = iree.do_not_optimize(%c2) : i32
  // CHECK-NEXT: return %[[C1]], %[[C2]]
  return %2, %3 : i32, i32
}

// -----

// CHECK-LABEL: @multiple_operands
func @multiple_operands() -> (i32, i32) {
  // CHECK-NEXT: %[[C1:.+]] = constant 1
  %c1 = constant 1 : i32
  // CHECK-NEXT: %[[C2:.+]] = constant 2
  %c2 = constant 2 : i32
  %0, %1 = iree.do_not_optimize(%c1, %c2) : i32, i32
  // CHECK-NEXT: return %[[C1]], %[[C2]]
  return %0, %1 : i32, i32
}

// -----

// CHECK-LABEL: @no_operands
func @no_operands() {
  iree.do_not_optimize()
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @no_fold_add
func @no_fold_add() -> (i32) {
  // CHECK-NEXT: %[[C1:.+]] = vm.const.i32 1 : i32
  %c1 = vm.const.i32 1 : i32
  %0 = iree.do_not_optimize(%c1) : i32
  // CHECK-NEXT: %[[R:.+]] = vm.add.i32 %[[C1]], %[[C1]]
  %1 = vm.add.i32 %0, %0 : i32
  // CHECK-NEXT: return %[[R]]
  return %1 : i32
}

// -----

// CHECK-LABEL: @deeply_nested
module @deeply_nested {
  // CHECK-LABEL: @middle
  module @middle {
    // CHECK-LABEL: @inner
    module @inner {
      // CHECK-LABEL: @no_operands
      func @no_operands() {
        iree.do_not_optimize()
        // CHECK-NEXT: return
        return
      }
    }
  }
}
