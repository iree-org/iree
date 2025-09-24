// RUN: iree-opt --split-input-file --iree-consteval-jit-debug --iree-consteval-jit-globals  %s | FileCheck %s
// XFAIL: *

// CHECK-LABEL: @eval_f64_scalar
// CHECK: 4.200000e+01 : f64
module @eval_i64_scalar {
  util.global private @offset : f64 = -2.0 : f64
  util.global private @hoisted : f64
  util.func public @main() -> f64 {
    %hoisted = util.global.load @hoisted : f64
    util.return %hoisted : f64
  }
  util.initializer {
    %cst = arith.constant 44.0 : f64
    %offset = util.global.load @offset : f64
    %sum = arith.addf %cst, %offset : f64
    util.global.store %sum, @hoisted : f64
    util.return
  }
}
