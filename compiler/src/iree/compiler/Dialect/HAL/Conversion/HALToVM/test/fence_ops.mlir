// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm %s | FileCheck %s

// CHECK-LABEL: @fence_create
// CHECK-SAME: (%[[SEMAPHORE0:.+]]: !vm.ref<!hal.semaphore>, %[[TIME0:.+]]: i64,
// CHECK-SAME:  %[[SEMAPHORE1:.+]]: !vm.ref<!hal.semaphore>, %[[TIME1:.+]]: i64)
func.func @fence_create(
    %semaphore0: !hal.semaphore, %time0: i64,
    %semaphore1: !hal.semaphore, %time1: i64) -> !hal.fence {
  // CHECK: %[[FENCE:.+]] = vm.call.variadic @hal.fence.create
  // CHECK-SAME: ([(%[[SEMAPHORE0]], %[[TIME0]]), (%[[SEMAPHORE1]], %[[TIME1]])])
  %fence = hal.fence.create
      at<%semaphore0 : !hal.semaphore>(%time0)
      at<%semaphore1 : !hal.semaphore>(%time1)
      -> !hal.fence
  // CHECK: vm.return %[[FENCE]]
  return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_join
// CHECK-SAME: (%[[FENCE0:.+]]: !vm.ref<!hal.fence>, %[[FENCE1:.+]]: !vm.ref<!hal.fence>)
func.func @fence_join(%fence0: !hal.fence, %fence1: !hal.fence) -> !hal.fence {
  // CHECK: %[[JOIN:.+]] = vm.call.variadic @hal.fence.join
  // CHECK-SAME: ([%[[FENCE0]], %[[FENCE1]]])
  %fence = hal.fence.join at([%fence0, %fence1]) -> !hal.fence
  // CHECK: vm.return %[[JOIN]]
  return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_signal
// CHECK-SAME: (%[[FENCE:.+]]: !vm.ref<!hal.fence>)
func.func @fence_signal(%fence: !hal.fence) {
  // CHECK: vm.call @hal.fence.signal(%[[FENCE]])
  hal.fence.signal<%fence : !hal.fence>
  // CHECK: vm.return
  return
}

// -----

// CHECK-LABEL: @fence_fail
// CHECK-SAME: (%[[FENCE:.+]]: !vm.ref<!hal.fence>, %[[STATUS:.+]]: i32)
func.func @fence_fail(%fence: !hal.fence, %status: i32) {
  // CHECK: vm.call @hal.fence.fail(%[[FENCE]], %[[STATUS]])
  hal.fence.fail<%fence : !hal.fence> status(%status)
  // CHECK: vm.return
  return
}

// -----

// CHECK-LABEL: @fence_await
// CHECK-SAME: (%[[FENCE0:.+]]: !vm.ref<!hal.fence>, %[[FENCE1:.+]]: !vm.ref<!hal.fence>,
// CHECK-SAME:  %[[TIMEOUT:.+]]: i32)
func.func @fence_await(%fence0: !hal.fence, %fence1: !hal.fence, %timeout: i32) -> i32 {
  // CHECK: %[[STATUS:.+]] = vm.call.variadic @hal.fence.await
  // CHECK-SAME: (%[[TIMEOUT]], [%[[FENCE0]], %[[FENCE1]]])
  %status = hal.fence.await until([%fence0, %fence1]) timeout_millis(%timeout) : i32
  // CHECK: vm.return %[[STATUS]]
  return %status : i32
}
