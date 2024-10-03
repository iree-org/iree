// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-util-test-conversion{widen-integers})' %s | FileCheck %s

// CHECK-LABEL: @assumeDivisibleOp
util.func public @assumeDivisibleOp(%arg0 : i16) -> i16 {
  // CHECK: util.assume.divisible {{.*}} by 4 : i32
  %0 = util.assume.divisible %arg0 by 4 : i16
  util.return %0 : i16
}

// -----
// CHECK-LABEL: @assumeNarrowOp
util.func public @assumeNarrowOp(%arg0 : i16) -> i16 {
  // CHECK: util.assume.narrow %arg0 : i32 to i8
  %0 = util.assume.narrow %arg0 : i16 to i8
  util.return %0 : i16
}

// -----
// CHECK-LABEL: @assumeRangeOp
util.func public @assumeRangeOp(%arg0 : i16) -> i16 {
  // CHECK: util.assume.range %arg0 in [4, 12] : i32
  %0 = util.assume.range %arg0 in [4, 12] : i16
  util.return %0 : i16
}

// -----
// CHECK-LABEL: @optimizationBarrier
util.func public @optimizationBarrier(%arg0 : i16) -> i16 {
  // CHECK: util.optimization_barrier %arg0 : i32
  %0 = util.optimization_barrier %arg0 : i16
  util.return %0 : i16
}
