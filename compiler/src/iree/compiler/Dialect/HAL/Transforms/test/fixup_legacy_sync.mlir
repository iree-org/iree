// RUN: iree-opt --split-input-file --iree-hal-fixup-legacy-sync %s | FileCheck %s

// Tests that command buffers that are reusable don't execute inline.
// Reusable + inline is not a valid combination.

module attributes {hal.device.targets = [#hal.device.target<"vulkan", {legacy_sync}>]} {
// CHECK-LABEL: @command_buffer_reusable
util.func public @command_buffer_reusable(%arg0: !hal.device) {
  // CHECK: hal.command_buffer.create device(%arg0 : !hal.device) mode("None")
  %cmd = hal.command_buffer.create device(%arg0 : !hal.device) mode("None") categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}
}  // module

// -----

// Tests that one-shot command buffers are allowed to execute inline.

module attributes {hal.device.targets = [#hal.device.target<"vulkan", {legacy_sync}>]} {
// CHECK-LABEL: @command_buffer_oneshot
util.func public @command_buffer_oneshot(%arg0: !hal.device) {
  // CHECK: hal.command_buffer.create device(%arg0 : !hal.device) mode("OneShot|AllowInlineExecution")
  %cmd = hal.command_buffer.create device(%arg0 : !hal.device) mode(OneShot) categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}
}  // module

// -----

// Tests for a no-op if there are no devices requiring legacy mode.

module attributes {hal.device.targets = [
  #hal.device.target<"vmvx", {}>,
  #hal.device.target<"vulkan", {}>
]} {
// CHECK-LABEL: @legacy_mode_not_required
util.func public @legacy_mode_not_required(%arg0: !hal.device) {
  // CHECK: hal.command_buffer.create device(%arg0 : !hal.device) mode(OneShot)
  %cmd = hal.command_buffer.create device(%arg0 : !hal.device) mode(OneShot) categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}
}  // module

// -----

// Tests that any device requiring legacy_sync will trigger the pass.

module attributes {hal.device.targets = [
  #hal.device.target<"vmvx", {}>,
  #hal.device.target<"vulkan", {legacy_sync}>
]} {
// CHECK-LABEL: @mixed_legacy_mode_required
util.func public @mixed_legacy_mode_required(%device: !hal.device, %wait: !hal.fence, %cmd: !hal.command_buffer, %signal: !hal.fence) {
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
}  // module

// -----

// Tests that queued operations get the appropriate waits before/after.

module attributes {hal.device.targets = [#hal.device.target<"vulkan", {legacy_sync}>]} {
// CHECK-LABEL: @blocking_execute
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[WAIT:.+]]: !hal.fence, %[[CMD:.+]]: !hal.command_buffer, %[[SIGNAL:.+]]: !hal.fence)
util.func public @blocking_execute(%device: !hal.device, %wait: !hal.fence, %cmd: !hal.command_buffer, %signal: !hal.fence) {
  %affinity = arith.constant 0 : i64
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
}  // module

// -----

// Tests that waits are not inserted if they already exist.

module attributes {hal.device.targets = [#hal.device.target<"vulkan", {legacy_sync}>]} {
// CHECK-LABEL: @blocking_execute
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[WAIT:.+]]: !hal.fence, %[[CMD:.+]]: !hal.command_buffer, %[[SIGNAL:.+]]: !hal.fence)
util.func public @blocking_execute(%device: !hal.device, %wait: !hal.fence, %cmd: !hal.command_buffer, %signal: !hal.fence) {
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
}  // module
