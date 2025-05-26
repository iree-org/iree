// RUN: iree-opt --split-input-file --iree-stream-elide-transfer-ops %s | FileCheck %s

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory}),
    (@dev_b -> @dev_a = {unified_memory}),
    (@dev_a -> @dev_c = {})
  ]>
} {

// Transfer between unified links should be elided.
// CHECK-LABEL: @elide_unified_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @elide_unified_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c128}

  // CHECK: util.return %[[RESOURCE]]
  util.return %transfer : !stream.resource<*>
}

// Transfer without explicit source should be elided.
// CHECK-LABEL: @elide_implicit_source_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @elide_implicit_source_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128}
      to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c128}

  // CHECK: util.return %[[RESOURCE]]
  util.return %transfer : !stream.resource<*>
}

// Transfer between non-unified links should not be elided.
// CHECK-LABEL: @preserve_non_unified_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @preserve_non_unified_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_c>) !stream.resource<*>{%c128}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

// Transfer that changes lifetime should not be elided.
// CHECK-LABEL: @preserve_lifetime_changing_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>)
util.func public @preserve_lifetime_changing_transfer(%resource: !stream.resource<transient>) -> !stream.resource<external> {
  %c128 = arith.constant 128 : index

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  %transfer = stream.async.transfer %resource : !stream.resource<transient>{%c128}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%c128}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

// Transfer with that is infered to be on the same device should be elided.
// CHECK-LABEL: @elide_same_device_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @elide_same_device_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128} -> !stream.resource<*>{%c128}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

// Explicit self-transfer on the same device should be elided.
// CHECK-LABEL: @elide_self_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @elide_self_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_a>) !stream.resource<*>{%c128}

  // CHECK: util.return %[[RESOURCE]]
  util.return %transfer : !stream.resource<*>
}

} // module
