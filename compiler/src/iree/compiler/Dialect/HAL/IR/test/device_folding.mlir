// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @ElideDeviceQueueBarrierOp
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:  %[[CMD:.+]]: !hal.command_buffer,
// CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence,
// CHECK-SAME:  %[[SIGNAL_FENCE:.+]]: !hal.fence)
func.func @ElideDeviceQueueBarrierOp(
    %device: !hal.device,
    %cmd: !hal.command_buffer,
    %wait_fence: !hal.fence,
    %signal_fence: !hal.fence) {
  %c-1_i64 = arith.constant -1 : i64

  // Temporary fence used to add an execution dependency.
  // Once the pattern runs this should be removed.
  // CHECK-NOT: hal.fence.create
  %fence0 = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence

  // Execute a command buffer sequencing %wait_fence -> %fence0.
  // The pattern should change this to signaling %signal_fence after removing
  // both of the barriers below.
  // CHECK: hal.device.queue.execute<%[[DEVICE]] : !hal.device>
  // CHECK-SAME: wait(%[[WAIT_FENCE]])
  // CHECK-SAME: signal(%[[SIGNAL_FENCE]])
  // CHECK-SAME: commands([%[[CMD]]])
  hal.device.queue.execute<%device : !hal.device>
      affinity(%c-1_i64)
      wait(%wait_fence)
      signal(%fence0)
      commands([%cmd])

  // CHECK-NOT: hal.fence.create
  %fence1 = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence

  // Execute a barrier sequencing %fence0 -> %fence1.
  // The pattern should remove this op.
  // CHECK-NOT: hal.device.queue.execute
  hal.device.queue.execute<%device : !hal.device>
      affinity(%c-1_i64)
      wait(%fence0)
      signal(%fence1)

  // Another op for the pattern to remove completing %fence1 -> %signal_fence
  // CHECK-NOT: hal.device.queue.execute
  hal.device.queue.execute<%device : !hal.device>
      affinity(%c-1_i64)
      wait(%fence1)
      signal(%signal_fence)

  // CHECK-NEXT: return
  return
}
