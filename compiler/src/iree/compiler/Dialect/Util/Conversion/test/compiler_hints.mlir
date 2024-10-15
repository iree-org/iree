// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-util-test-conversion{widen-integers})' %s | FileCheck %s

// -----
// CHECK-LABEL: @assumeIntOp
util.func public @assumeIntOp(%arg0 : i16) -> i16 {
  // CHECK: util.assume.int %arg0<umin = 1> : i32
  %0 = util.assume.int %arg0<umin = 1> : i16
  util.return %0 : i16
}

// -----
// CHECK-LABEL: @optimizationBarrier
util.func public @optimizationBarrier(%arg0 : i16) -> i16 {
  // CHECK: util.optimization_barrier %arg0 : i32
  %0 = util.optimization_barrier %arg0 : i16
  util.return %0 : i16
}
