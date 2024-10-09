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
