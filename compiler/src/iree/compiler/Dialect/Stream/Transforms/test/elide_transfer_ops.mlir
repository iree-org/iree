// RUN: iree-opt --split-input-file --iree-stream-elide-transfer-ops %s | FileCheck %s

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    {[@dev_a, @dev_b], transfer_required = false},
    {[@dev_a, @dev_c], transfer_required = true}
  ]>
} {

// Transfer between unified links should be elided
// CHECK-LABEL: @elide_unified_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @elide_unified_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128}
      from(#hal.device.affinity<@dev_a>) -> to(#hal.device.affinity<@dev_b>) !stream.resource<*>{%c128}

  // CHECK: util.return %[[RESOURCE]]
  util.return %transfer : !stream.resource<*>
}

// Transfer between non-unified links should not be elided
// CHECK-LABEL: @preserve_non_unified_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @preserve_non_unified_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128}
      from(#hal.device.affinity<@dev_a>) -> to(#hal.device.affinity<@dev_c>) !stream.resource<*>{%c128}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

// Transfer that changes lifetime should not be elided
// CHECK-LABEL: @preserve_lifetime_changing_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>)
util.func public @preserve_lifetime_changing_transfer(%resource: !stream.resource<transient>) -> !stream.resource<external> {
  %c128 = arith.constant 128 : index

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  %transfer = stream.async.transfer %resource : !stream.resource<transient>{%c128}
      from(#hal.device.affinity<@dev_a>) -> to(#hal.device.affinity<@dev_a>) !stream.resource<external>{%c128}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

// Transfer with missing affinity information should not be elided
// CHECK-LABEL: @preserve_missing_affinity
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @preserve_missing_affinity(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128} -> !stream.resource<*>{%c128}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

// Self-transfer on the same device should be elided
// CHECK-LABEL: @elide_self_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @elide_self_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index

  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%c128}
      from(#hal.device.affinity<@dev_a>) -> to(#hal.device.affinity<@dev_a>) !stream.resource<*>{%c128}

  // CHECK: util.return %[[RESOURCE]]
  util.return %transfer : !stream.resource<*>
}

} // module
