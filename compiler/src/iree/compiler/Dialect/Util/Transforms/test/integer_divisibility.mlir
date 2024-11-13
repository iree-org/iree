// RUN: iree-opt --split-input-file --iree-util-optimize-int-arithmetic  %s | FileCheck %s

// Use the int arithmetic optimization pipeline to test the integer divisibility
// analysis. This largely relies on the arith.remui operation resolving to a
// constant 0 on even division.

// CHECK-LABEL: @remui_div_by_exact_factor
util.func @remui_div_by_exact_factor(%arg0 : index) -> index {
  %cst = arith.constant 16 : index
  %0 = util.assume.int %arg0<udiv = 16> : index
  %1 = arith.remui %0, %cst : index
  // CHECK: %[[CST:.*]] = arith.constant 0
  // CHECK: return %[[CST]]
  util.return %1 : index
}

// -----
// CHECK-LABEL: @remui_div_by_common_factor
util.func @remui_div_by_common_factor(%arg0 : index) -> index {
  %cst = arith.constant 8 : index
  %0 = util.assume.int %arg0<udiv = 16> : index
  %1 = arith.remui %0, %cst : index
  // CHECK: %[[CST:.*]] = arith.constant 0
  // CHECK: return %[[CST]]
  util.return %1 : index
}

// -----
// CHECK-LABEL: @remui_div_by_unrelated
util.func @remui_div_by_unrelated(%arg0 : index) -> index {
  %cst = arith.constant 23 : index
  %0 = util.assume.int %arg0<udiv = 16> : index
  // CHECK: arith.remui
  %1 = arith.remui %0, %cst : index
  util.return %1 : index
}

// -----
// A missing udiv in a multi-row assumption is treated as an unknown.
// CHECK-LABEL: @missing_udiv_skipped
util.func @missing_udiv_skipped(%arg0 : index) -> index {
  // CHECK: arith.remui
  %cst = arith.constant 16 : index
  %0 = util.assume.int %arg0[<udiv = 16>, <>] : index
  %1 = arith.remui %0, %cst : index
  util.return %1 : index
}

// -----

util.func @muli_divisibility(%arg0 : index) -> (index, index) {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %0 = arith.muli %arg0, %c16 : index
  %1 = arith.remui %0, %c16 : index
  %2 = arith.remui %0, %c32 : index
  util.return %1, %2 : index, index
}
// CHECK-LABEL: @muli_divisibility
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//       CHECK:   %[[V:.+]] = arith.muli
//       CHECK:   %[[REM:.+]] = arith.remui %[[V]], %[[C32]]
//       CHECK:   return %[[C0]], %[[REM]]

// -----

util.func @muli_compounded_divisibility(%arg0 : index) -> (index, index) {
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %0 = util.assume.int %arg0<udiv = 4> : index
  %1 = arith.muli %0, %c16 : index
  %2 = arith.remui %1, %c64 : index
  %3 = arith.remui %1, %c128 : index
  util.return %2, %3 : index, index
}
// CHECK-LABEL: @muli_compounded_divisibility
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//       CHECK:   %[[V:.+]] = arith.muli
//       CHECK:   %[[REM:.+]] = arith.remui %[[V]], %[[C128]]
//       CHECK:   return %[[C0]], %[[REM]]

// -----

util.func @divui_divisibility(%arg0 : index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %0 = util.assume.int %arg0<udiv = 64> : index
  %1 = arith.divui %0, %c4 : index
  %2 = arith.remui %1, %c16 : index
  %3 = arith.remui %1, %c32 : index
  util.return %2, %3 : index, index
}
// CHECK-LABEL: @divui_divisibility
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//       CHECK:   %[[V:.+]] = arith.divui
//       CHECK:   %[[REM:.+]] = arith.remui %[[V]], %[[C32]]
//       CHECK:   return %[[C0]], %[[REM]]
