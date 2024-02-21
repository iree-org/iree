// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm %s | FileCheck %s

// CHECK-LABEL: @fence_create
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
util.func public @fence_create(%device: !hal.device) -> !hal.fence {
  // CHECK: %[[FLAGS:.+]] = vm.const.i32.zero
  // CHECK: %[[FENCE:.+]] = vm.call @hal.fence.create(%[[DEVICE]], %[[FLAGS]])
  %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
  // CHECK: vm.return %[[FENCE]]
  util.return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_join
// CHECK-SAME: (%[[FENCE0:.+]]: !vm.ref<!hal.fence>, %[[FENCE1:.+]]: !vm.ref<!hal.fence>)
util.func public @fence_join(%fence0: !hal.fence, %fence1: !hal.fence) -> !hal.fence {
  // CHECK: %[[JOIN:.+]] = vm.call.variadic @hal.fence.join
  // CHECK-SAME: ([%[[FENCE0]], %[[FENCE1]]])
  %fence = hal.fence.join at([%fence0, %fence1]) -> !hal.fence
  // CHECK: vm.return %[[JOIN]]
  util.return %fence : !hal.fence
}

// -----

// CHECK-LABEL: @fence_query
// CHECK-SAME: (%[[FENCE:.+]]: !vm.ref<!hal.fence>)
util.func public @fence_query(%fence: !hal.fence) -> i32 {
  // CHECK: %[[STATUS:.+]] = vm.call @hal.fence.query(%[[FENCE]])
  %status = hal.fence.query<%fence : !hal.fence> : i32
  // CHECK: vm.return %[[STATUS]]
  util.return %status : i32
}

// -----

// CHECK-LABEL: @fence_signal
// CHECK-SAME: (%[[FENCE:.+]]: !vm.ref<!hal.fence>)
util.func public @fence_signal(%fence: !hal.fence) {
  // CHECK: vm.call @hal.fence.signal(%[[FENCE]])
  hal.fence.signal<%fence : !hal.fence>
  // CHECK: vm.return
  util.return
}

// -----

// CHECK-LABEL: @fence_fail
// CHECK-SAME: (%[[FENCE:.+]]: !vm.ref<!hal.fence>, %[[STATUS:.+]]: i32)
util.func public @fence_fail(%fence: !hal.fence, %status: i32) {
  // CHECK: vm.call @hal.fence.fail(%[[FENCE]], %[[STATUS]])
  hal.fence.fail<%fence : !hal.fence> status(%status)
  // CHECK: vm.return
  util.return
}

// -----

// CHECK-LABEL: @fence_await
// CHECK-SAME: (%[[FENCE0:.+]]: !vm.ref<!hal.fence>, %[[FENCE1:.+]]: !vm.ref<!hal.fence>,
// CHECK-SAME:  %[[TIMEOUT:.+]]: i32)
util.func public @fence_await(%fence0: !hal.fence, %fence1: !hal.fence, %timeout: i32) -> i32 {
  // CHECK: %[[STATUS:.+]] = vm.call.variadic @hal.fence.await
  // CHECK-SAME: (%[[TIMEOUT]], [%[[FENCE0]], %[[FENCE1]]])
  %status = hal.fence.await until([%fence0, %fence1]) timeout_millis(%timeout) : i32
  // CHECK: vm.return %[[STATUS]]
  util.return %status : i32
}
