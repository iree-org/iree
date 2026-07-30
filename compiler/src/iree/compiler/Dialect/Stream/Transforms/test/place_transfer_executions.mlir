// RUN: iree-opt --split-input-file --iree-stream-place-transfer-executions %s | FileCheck %s

module {
  util.global private @device_a : !hal.device
  util.global private @device_b : !hal.device

  // CHECK-LABEL: util.func public @pinned_transfer
  // CHECK: stream.async.transfer on(#hal.device.affinity<@device_a>)
  // CHECK-SAME: from(#hal.device.affinity<@device_b>)
  // CHECK-SAME: to(#hal.device.affinity<@device_a>)
  util.func @pinned_transfer(%source: !stream.resource<transient>, %size: index) -> !hal.buffer_view {
    %result = stream.async.transfer %source : !stream.resource<transient>{%size} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_a>) !stream.resource<external>{%size}
    %export = stream.tensor.export on(#hal.device.affinity<@device_a>) %result : tensor<?xi8>{%size} in !stream.resource<external>{%size} -> !hal.buffer_view
    util.return %export : !hal.buffer_view
  }
}

// -----

module {
  util.global private @device_a : !hal.device
  util.global private @device_b : !hal.device

  // CHECK-LABEL: util.func public @unpinned_transfer
  // CHECK-NOT: stream.async.transfer on(
  // CHECK: stream.async.transfer
  util.func @unpinned_transfer(%source: !stream.resource<transient>, %size: index) -> !stream.resource<external> {
    %result = stream.async.transfer %source : !stream.resource<transient>{%size} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_a>) !stream.resource<external>{%size}
    util.return %result : !stream.resource<external>
  }
}

// -----

module {
  util.global private @device_a : !hal.device

  // CHECK-LABEL: util.func public @compatible_transfer
  // CHECK-NOT: stream.async.transfer on(
  // CHECK: stream.async.transfer
  util.func @compatible_transfer(%source: !stream.resource<transient>, %size: index) -> !hal.buffer_view {
    %result = stream.async.transfer %source : !stream.resource<transient>{%size} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<external>{%size}
    %export = stream.tensor.export on(#hal.device.affinity<@device_a>) %result : tensor<?xi8>{%size} in !stream.resource<external>{%size} -> !hal.buffer_view
    util.return %export : !hal.buffer_view
  }
}
