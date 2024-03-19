// RUN: iree-opt --split-input-file --iree-util-drop-compiler-hints %s | FileCheck --implicit-check-not="util.optimization_barrier" %s

// This file is used as an example in docs/developing_iree/developer_overview.md.
// If you move or delete it, please update the documentation accordingly.

// CHECK-LABEL: @constant
util.func @constant() -> i32 {
  // CHECK-NEXT: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  %0 = util.optimization_barrier %c1 : i32
  // CHECK-NEXT: util.return %[[C1]]
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @multiple
util.func @multiple() -> (i32, i32) {
  // CHECK-NEXT: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  %0 = util.optimization_barrier %c1 : i32
  %1 = util.optimization_barrier %0 : i32
  // CHECK-NEXT: %[[C2:.+]] = arith.constant 2
  %c2 = arith.constant 2 : i32
  %2 = util.optimization_barrier %1 : i32
  %3 = util.optimization_barrier %c2 : i32
  // CHECK-NEXT: util.return %[[C1]], %[[C2]]
  util.return %2, %3 : i32, i32
}

// -----

// CHECK-LABEL: @multiple_operands
util.func @multiple_operands() -> (i32, i32) {
  // CHECK-NEXT: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  // CHECK-NEXT: %[[C2:.+]] = arith.constant 2
  %c2 = arith.constant 2 : i32
  %0, %1 = util.optimization_barrier %c1, %c2 : i32, i32
  // CHECK-NEXT: util.return %[[C1]], %[[C2]]
  util.return %0, %1 : i32, i32
}

// -----

// CHECK-LABEL: @no_fold_add
util.func @no_fold_add() -> (i32) {
  // CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : i32
  %c1 = arith.constant 1 : i32
  %0 = util.optimization_barrier %c1 : i32
  // CHECK-NEXT: %[[R:.+]] = arith.addi %[[C1]], %[[C1]]
  %1 = arith.addi %0, %0 : i32
  // CHECK-NEXT: util.return %[[R]]
  util.return %1 : i32
}

// -----

// CHECK-LABEL: @deeply_nested
module @deeply_nested {
  // CHECK-LABEL: @middle
  module @middle {
    // CHECK-LABEL: @inner
    module @inner {
      // CHECK-LABEL: @constant
      util.func @constant() -> i32 {
        // CHECK-NEXT: %[[C1:.+]] = arith.constant 1
        %c1 = arith.constant 1 : i32
        %0 = util.optimization_barrier %c1 : i32
        // CHECK-NEXT: util.return %[[C1]]
        util.return %0 : i32
      }
    }
  }
}
