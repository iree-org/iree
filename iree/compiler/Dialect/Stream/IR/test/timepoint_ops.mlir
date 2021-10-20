// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @timepointImmediate
func @timepointImmediate() -> !stream.timepoint {
  // CHECK: = stream.timepoint.immediate => !stream.timepoint
  %0 = stream.timepoint.immediate => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointJoin
func @timepointJoin(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: = stream.timepoint.join max(%arg0, %arg1) => !stream.timepoint
  %0 = stream.timepoint.join max(%arg0, %arg1) => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointAwait
func @timepointAwait(%arg0: !stream.timepoint, %arg1: !stream.resource<staging>, %arg2: !stream.resource<*>) -> (!stream.resource<staging>, !stream.resource<*>) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  %0:2 = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  return %0#0, %0#1 : !stream.resource<staging>, !stream.resource<*>
}
