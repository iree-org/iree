// RUN: iree-opt --split-input-file --iree-hal-fixup-legacy-sync %s | FileCheck %s

// TODO(multi-device): remove once device globals are used. This is a fallback
// path during the transition.
module attributes {hal.device.targets = [
  #hal.device.target<"vulkan", {legacy_sync}>
]} {
// CHECK-LABEL: @default_device_targets
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
util.func public @default_device_targets(%device: !hal.device) {
  // CHECK: hal.command_buffer.create device(%[[DEVICE]] : !hal.device) mode("None")
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("None") categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}
}  // module

// -----

// Tests that unknown devices (here passed as an arg on a public function)
// don't trigger the pass, as we default to non-legacy behavior.

// CHECK-LABEL: @unknown_device
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
util.func public @unknown_device(%device: !hal.device) {
  // CHECK: hal.command_buffer.create device(%[[DEVICE]] : !hal.device) mode("None")
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("None") categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}

// -----

// Tests that command buffers that are reusable don't execute inline.
// Reusable + inline is not a valid combination.

util.global private @device = #hal.device.target<"vulkan", {legacy_sync}> : !hal.device

// CHECK-LABEL: @command_buffer_reusable
util.func public @command_buffer_reusable() {
  // CHECK: %[[DEVICE:.+]] = util.global.load @device
  %device = util.global.load @device : !hal.device
  // CHECK: hal.command_buffer.create device(%[[DEVICE]] : !hal.device) mode("None")
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("None") categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}

// -----

// Tests that one-shot command buffers are allowed to execute inline.

util.global private @device = #hal.device.target<"vulkan", {legacy_sync}> : !hal.device

// CHECK-LABEL: @command_buffer_oneshot
util.func public @command_buffer_oneshot() {
  // CHECK: %[[DEVICE:.+]] = util.global.load @device
  %device = util.global.load @device : !hal.device
  // CHECK: hal.command_buffer.create device(%[[DEVICE]] : !hal.device) mode("OneShot|AllowInlineExecution")
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode(OneShot) categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}

// -----

// Tests for a no-op if there are no devices requiring legacy mode.

util.global private @device = #hal.device.select<[
  #hal.device.target<"local", {}>,
  #hal.device.target<"vulkan", {}>
]> : !hal.device

// CHECK-LABEL: @legacy_mode_not_required
util.func public @legacy_mode_not_required() {
  // CHECK: %[[DEVICE:.+]] = util.global.load @device
  %device = util.global.load @device : !hal.device
  // CHECK: hal.command_buffer.create device(%[[DEVICE]] : !hal.device) mode(OneShot)
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode(OneShot) categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}

// -----

// Tests that any device requiring legacy_sync in a set will trigger the pass.

util.global private @device = #hal.device.select<[
  #hal.device.target<"local", {}>,
  #hal.device.target<"vulkan", {legacy_sync}>
]> : !hal.device

// CHECK-LABEL: @mixed_legacy_mode_required
util.func public @mixed_legacy_mode_required(%wait: !hal.fence, %cmd: !hal.command_buffer, %signal: !hal.fence) {
  %device = util.global.load @device : !hal.device
  %affinity = arith.constant 0 : i64
  // CHECK: hal.fence.await
  // CHECK: hal.device.queue.execute
  // CHECK: hal.fence.await
  hal.device.queue.execute<%device : !hal.device>
      affinity(%affinity)
      wait(%wait) signal(%signal)
      commands([%cmd])
  util.return
}

// -----

// Tests that only devices with legacy_sync trigger the pass.

util.global private @device_async = #hal.device.target<"local", {}> : !hal.device
util.global private @device_sync = #hal.device.target<"vulkan", {legacy_sync}> : !hal.device

// CHECK-LABEL: @mixed_legacy_mode_scoped
util.func public @mixed_legacy_mode_scoped(%wait: !hal.fence, %cmd: !hal.command_buffer, %signal: !hal.fence) {
  // CHECK-DAG: %[[DEVICE_ASYNC:.+]] = util.global.load @device_async
  %device_async = util.global.load @device_async : !hal.device
  // CHECK-DAG: %[[DEVICE_SYNC:.+]] = util.global.load @device_sync
  %device_sync = util.global.load @device_sync : !hal.device
  %affinity = arith.constant 0 : i64
  // CHECK-NOT: hal.fence.await
  // CHECK: hal.device.queue.execute<%[[DEVICE_ASYNC]]
  // CHECK-NOT: hal.fence.await
  hal.device.queue.execute<%device_async : !hal.device>
      affinity(%affinity)
      wait(%wait) signal(%signal)
      commands([%cmd])
  // CHECK: hal.fence.await
  // CHECK: hal.device.queue.execute<%[[DEVICE_SYNC]]
  // CHECK: hal.fence.await
  hal.device.queue.execute<%device_sync : !hal.device>
      affinity(%affinity)
      wait(%wait) signal(%signal)
      commands([%cmd])
  util.return
}

// -----

// Tests that queued operations get the appropriate waits before/after.

util.global private @device = #hal.device.target<"vulkan", {legacy_sync}> : !hal.device

// CHECK-LABEL: @blocking_execute
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[CMD:.+]]: !hal.command_buffer, %[[SIGNAL:.+]]: !hal.fence)
util.func public @blocking_execute(%wait: !hal.fence, %cmd: !hal.command_buffer, %signal: !hal.fence) {
  %affinity = arith.constant 0 : i64
  // CHECK: %[[DEVICE:.+]] = util.global.load @device
  %device = util.global.load @device : !hal.device
  //  CHECK-DAG: %[[NULL:.+]] = util.null : !hal.fence
  //  CHECK-DAG: hal.fence.await until([%[[WAIT]]])
  // CHECK-NEXT: hal.device.queue.execute<%[[DEVICE]] : !hal.device>
  // CHECK-SAME:   wait(%[[NULL]]) signal(%[[SIGNAL]])
  // CHECK-SAME:   commands([%[[CMD]]])
  // CHECK-NEXT: hal.fence.await until([%[[SIGNAL]]])
  hal.device.queue.execute<%device : !hal.device>
      affinity(%affinity)
      wait(%wait) signal(%signal)
      commands([%cmd])
  util.return
}

// -----

// Tests that waits are not inserted if they already exist.

util.global private @device = #hal.device.target<"vulkan", {legacy_sync}> : !hal.device

// CHECK-LABEL: @blocking_execute
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[CMD:.+]]: !hal.command_buffer, %[[SIGNAL:.+]]: !hal.fence)
util.func public @blocking_execute(%wait: !hal.fence, %cmd: !hal.command_buffer, %signal: !hal.fence) {
  // CHECK: %[[DEVICE:.+]] = util.global.load @device
  %device = util.global.load @device : !hal.device
  // CHECK-NEXT: %[[TIMEOUT:.+]] = arith.constant 100
  %timeout = arith.constant 100 : i32
  // CHECK-NEXT: hal.fence.await until([%[[WAIT]]]) timeout_millis(%[[TIMEOUT]])
  hal.fence.await until([%wait]) timeout_millis(%timeout) : i32
  // This should not block the search:
  // CHECK-NEXT: arith.constant 0
  %affinity = arith.constant 0 : i64
  // CHECK-NEXT: %[[NULL:.+]] = util.null : !hal.fence
  // CHECK-NEXT: hal.device.queue.execute<%[[DEVICE]] : !hal.device>
  // CHECK-SAME:   wait(%[[NULL]]) signal(%[[SIGNAL]])
  // CHECK-SAME:   commands([%[[CMD]]])
  hal.device.queue.execute<%device : !hal.device>
      affinity(%affinity)
      wait(%wait) signal(%signal)
      commands([%cmd])
  // CHECK-NEXT: hal.fence.await until([%[[SIGNAL]]]) timeout_millis(%[[TIMEOUT]])
  hal.fence.await until([%signal]) timeout_millis(%timeout) : i32
  // CHECK-NEXT: util.return
  util.return
}
