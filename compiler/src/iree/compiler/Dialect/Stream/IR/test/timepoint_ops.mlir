// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @timepointImmediate
func.func @timepointImmediate() -> !stream.timepoint {
  // CHECK: = stream.timepoint.immediate => !stream.timepoint
  %0 = stream.timepoint.immediate => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointImport
func.func @timepointImport(%arg0: !hal.semaphore, %arg1: index) -> !stream.timepoint {
  // CHECK: = stream.timepoint.import %arg0, %arg1 : (!hal.semaphore, index) => !stream.timepoint
  %0 = stream.timepoint.import %arg0, %arg1 : (!hal.semaphore, index) => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointExport
func.func @timepointExport(%arg0: !stream.timepoint) -> (!hal.semaphore, index) {
  // CHECK: = stream.timepoint.export %arg0 => (!hal.semaphore, index)
  %0:2 = stream.timepoint.export %arg0 => (!hal.semaphore, index)
  return %0#0, %0#1 : !hal.semaphore, index
}

// -----

// CHECK-LABEL: @timepointChainExternal
func.func @timepointChainExternal(%arg0: !stream.timepoint, %arg1: !hal.fence) {
  // CHECK: stream.timepoint.chain_external %arg0 => (%arg1 : !hal.fence)
  stream.timepoint.chain_external %arg0 => (%arg1 : !hal.fence)
  return
}

// -----

// CHECK-LABEL: @timepointJoin
func.func @timepointJoin(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: = stream.timepoint.join max(%arg0, %arg1) => !stream.timepoint
  %0 = stream.timepoint.join max(%arg0, %arg1) => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointAwait
func.func @timepointAwait(%arg0: !stream.timepoint, %arg1: !stream.resource<staging>, %arg2: !stream.resource<*>) -> (!stream.resource<staging>, !stream.resource<*>) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  %0:2 = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  return %0#0, %0#1 : !stream.resource<staging>, !stream.resource<*>
}
