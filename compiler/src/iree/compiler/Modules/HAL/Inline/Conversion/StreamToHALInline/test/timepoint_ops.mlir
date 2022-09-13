// RUN: iree-opt --split-input-file --iree-hal-inline-conversion %s | FileCheck %s

// NOTE: the inline HAL doesn't model timepoints and we just turn them into ints
// that'll eventually get DCE'd.

// CHECK-LABEL: @rwTimepoint
// CHECK-SAME: = 0 : i64
util.global private mutable @rwTimepoint = #stream.timepoint<immediate>
// CHECK: func.func @globalTimepoint(%arg0: i64) -> i64
func.func @globalTimepoint(%arg0: !stream.timepoint) -> !stream.timepoint {
  // CHECK: util.global.store %arg0, @rwTimepoint
  util.global.store %arg0, @rwTimepoint : !stream.timepoint
  // CHECK: %[[VALUE:.+]] = util.global.load @rwTimepoint
  %value = util.global.load @rwTimepoint : !stream.timepoint
  // CHECK: return %[[VALUE]]
  return %value : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointImmediate
func.func @timepointImmediate() -> !stream.timepoint {
  // CHECK: %[[TIMEPOINT:.+]] = arith.constant 0
  %0 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: return %[[TIMEPOINT]]
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointJoin
func.func @timepointJoin(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[TIMEPOINT:.+]] = arith.constant 0
  %0 = stream.timepoint.join max(%arg0, %arg1) => !stream.timepoint
  // CHECK: return %[[TIMEPOINT]]
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointAwait
func.func @timepointAwait(%arg0: !stream.timepoint, %arg1: !stream.resource<staging>, %arg2: !stream.resource<*>) -> (!stream.resource<staging>, !stream.resource<*>) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %0:2 = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  // CHECK: return %arg1, %arg2
  return %0#0, %0#1 : !stream.resource<staging>, !stream.resource<*>
}
