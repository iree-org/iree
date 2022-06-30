// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// TODO(#1285): implement timepoint lowering into HAL semaphores.
// For now all timepoints turn into ints and are mostly ignored.

// CHECK-LABEL: @rwTimepoint
// CHECK-SAME: = 0 : index
util.global private mutable @rwTimepoint = #stream.timepoint<immediate>
// CHECK: func.func @globalTimepoint(%arg0: index) -> index
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

// CHECK-LABEL: @timepointImport
func.func @timepointImport(%arg0: !hal.semaphore, %arg1: index) -> !stream.timepoint {
  // CHECK: %[[WAIT_OK:.+]] = hal.semaphore.await<%arg0 : !hal.semaphore> until(%arg1) : i32
  // CHECK: util.status.check_ok %[[WAIT_OK]]
  // CHECK: %[[TIMEPOINT:.+]] = arith.constant 0
  %0 = stream.timepoint.import %arg0, %arg1 : (!hal.semaphore, index) => !stream.timepoint
  // CHECK: return %[[TIMEPOINT]]
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointExport
func.func @timepointExport(%arg0: !stream.timepoint) -> (!hal.semaphore, index) {
  // CHECK: %[[TIMEPOINT:.+]] = arith.constant 0
  // CHECK: %[[SEMAPHORE:.+]] = hal.semaphore.create device(%device : !hal.device) initial(%c0) : !hal.semaphore
  %0:2 = stream.timepoint.export %arg0 => (!hal.semaphore, index)
  // CHECK: return %[[SEMAPHORE]], %[[TIMEPOINT]]
  return %0#0, %0#1 : !hal.semaphore, index
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
