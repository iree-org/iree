// RUN: iree-opt --iree-stream-transformation-pipeline %s | FileCheck %s

// End-to-end regression for https://github.com/iree-org/iree/issues/21361.
// The final result is pinned to device A but produced by a dispatch on device
// B. The transfer into the external result must execute on A so that the final
// allocation remains pinned to A. Only the transient feeding that transfer
// needs to be accessible from both devices.

module {
  util.global private @device_a : !hal.device
  util.global private @device_b : !hal.device

  // CHECK-LABEL: util.func public @multi_device_mul
  util.func public @multi_device_mul(%input: !hal.buffer_view) -> !hal.buffer_view {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index

    %import = stream.tensor.import on(#hal.device.affinity<@device_a>) %input : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %produced_a = stream.async.dispatch on(#hal.device.affinity<@device_a>) @executable::@dispatch_a(%import[%c0 to %c16 for %c16]) : (!stream.resource<external>{%c16}) -> !stream.resource<transient>{%c16}
    %transferred_b = stream.async.transfer %produced_a : !stream.resource<transient>{%c16} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%c16}

    // The dispatch result produced on B and read by the transfer on A must be
    // accessible from both devices.
    // CHECK: %[[B_RESULT:.+]], %[[B_RESULT_ALLOCATED:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>) await(%[[A_DONE:.+]])
    // CHECK: %[[B_DONE:.+]] = stream.cmd.execute on(#hal.device.affinity<@device_b>)
    // CHECK-SAME: %[[B_RESULT]] as %[[B_RESULT_CAPTURE:[a-zA-Z0-9_]+]]: !stream.resource<transient>
    // CHECK: stream.cmd.dispatch @executable::@dispatch_b
    // CHECK: wo %[[B_RESULT_CAPTURE]]
    %produced_b = stream.async.dispatch on(#hal.device.affinity<@device_b>) @executable::@dispatch_b(%transferred_b[%c0 to %c16 for %c16]) : (!stream.resource<transient>{%c16}) -> !stream.resource<transient>{%c16}

    // The final external allocation remains pinned to A. The transfer is a
    // separate execution on A that makes the B-written intermediate visible
    // and copies it into the pinned allocation.
    // CHECK: %[[FINAL:.+]], %[[FINAL_ALLOCATED:.+]] = stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>){{.*}} => !stream.resource<external>
    // CHECK: %[[TRANSFER_DONE:.+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>)
    // CHECK-SAME: %[[B_RESULT]] as %[[TRANSFER_SOURCE:[a-zA-Z0-9_]+]]: !stream.resource<transient>
    // CHECK-SAME: %[[FINAL]] as %[[TRANSFER_TARGET:[a-zA-Z0-9_]+]]: !stream.resource<external>
    // CHECK: stream.cmd.invalidate from(#hal.device.affinity<@device_b>) %[[TRANSFER_SOURCE]]
    // CHECK-NEXT: stream.cmd.copy %[[TRANSFER_SOURCE]]{{.*}}, %[[TRANSFER_TARGET]]
    %result = stream.async.transfer %produced_b : !stream.resource<transient>{%c16} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_a>) !stream.resource<external>{%c16}

    // CHECK: %[[AWAITED:.+]] = stream.timepoint.await {{.*}} => %[[FINAL]]
    // CHECK: stream.tensor.export on(#hal.device.affinity<@device_a>) %[[AWAITED]]
    %export = stream.tensor.export on(#hal.device.affinity<@device_a>) %result : tensor<4xf32> in !stream.resource<external>{%c16} -> !hal.buffer_view
    util.return %export : !hal.buffer_view
  }
}
