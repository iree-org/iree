// RUN: iree-opt --split-input-file --iree-hal-materialize-timelines %s | FileCheck %s

// CHECK: util.global private @_timeline_semaphore : !hal.semaphore
// CHECK: util.initializer {
// CHECK:   %[[DEVICE:.+]] = hal.ex.shared_device
// CHECK:   %[[SEMAPHORE:.+]] = hal.semaphore.create
// CHECK-SAME: device(%[[DEVICE]] : !hal.device)
// CHECK-SAME: initial(%c0_i64)
// CHECK-NEXT: util.global.store %[[SEMAPHORE]], @_timeline_semaphore
// CHECK: }

// CHECK: util.global private mutable @_timeline_value = 0 : i64

// CHECK-LABEL: @fn1
func.func @fn1() -> !hal.fence {
  // CHECK: %[[SEMAPHORE:.+]] = util.global.load @_timeline_semaphore
  // CHECK: %[[CURRENT_VALUE:.+]] = util.global.load @_timeline_value
  // CHECK: %[[NEXT_VALUE:.+]] = arith.addi %[[CURRENT_VALUE]], %c1
  // CHECK: util.global.store %[[NEXT_VALUE]], @_timeline_value
  // CHECK: %[[FENCE0:.+]] = hal.fence.create at<%[[SEMAPHORE]] : !hal.semaphore>(%[[NEXT_VALUE]])
  %0 = hal.timeline.advance : !hal.fence
  // CHECK: return %[[FENCE0]]
  return %0 : !hal.fence
}

// CHECK-LABEL: @fn2
func.func @fn2(%arg0: i1, %arg1: !hal.fence) -> !hal.fence {
  // CHECK: %[[FENCE:.+]] = scf.if
  %0 = scf.if %arg0 -> (!hal.fence) {
    // CHECK: scf.yield %arg1
    scf.yield %arg1 : !hal.fence
  } else {
    // CHECK: %[[SEMAPHORE:.+]] = util.global.load @_timeline_semaphore
    // CHECK: %[[CURRENT_VALUE:.+]] = util.global.load @_timeline_value
    // CHECK: %[[NEXT_VALUE:.+]] = arith.addi %[[CURRENT_VALUE]], %c1
    // CHECK: util.global.store %[[NEXT_VALUE]], @_timeline_value
    // CHECK: %[[NEW_FENCE:.+]] = hal.fence.create at<%[[SEMAPHORE]] : !hal.semaphore>(%[[NEXT_VALUE]])
    %1 = hal.timeline.advance : !hal.fence
    // CHECK: scf.yield %[[NEW_FENCE]]
    scf.yield %1 : !hal.fence
  }
  // CHECK: return %[[FENCE]]
  return %0 : !hal.fence
}
