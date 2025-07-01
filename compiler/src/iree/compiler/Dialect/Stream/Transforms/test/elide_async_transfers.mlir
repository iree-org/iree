// RUN: iree-opt --split-input-file --iree-stream-elide-async-transfers %s | FileCheck %s

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

// Transfer between unified links should be elided.
// CHECK-LABEL: @elide_unified_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @elide_unified_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %size = arith.constant 4 : index

  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%size}

  // CHECK: util.return %[[RESOURCE]]
  util.return %transfer : !stream.resource<*>
}

} // module

// -----

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

// Transfer without explicit source should be elided.
// CHECK-LABEL: @elide_implicit_source_transfer
// CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !hal.buffer_view)
util.func public @elide_implicit_source_transfer(%buffer_view: !hal.buffer_view) -> !stream.resource<external> {
  %size = arith.constant 4 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %buffer_view : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%size}
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %imported : !stream.resource<external>{%size} ->
      to(#hal.device.promise<@dev_b>) !stream.resource<external>{%size}

  // CHECK: util.return %[[IMPORT]]
  util.return %transfer : !stream.resource<external>
}

} // module

// -----

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_c = {})
  ]>
} {

// Transfer between non-unified links should not be elided.
// CHECK-LABEL: @preserve_non_unified_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @preserve_non_unified_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %size = arith.constant 4 : index

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_c>) !stream.resource<*>{%size}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

} // module

// -----

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

// Transfer that changes lifetime should not be elided.
// CHECK-LABEL: @preserve_lifetime_changing_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>)
util.func public @preserve_lifetime_changing_transfer(%resource: !stream.resource<transient>) -> !stream.resource<external> {
  %size = arith.constant 4 : index

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  %transfer = stream.async.transfer %resource : !stream.resource<transient>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_a>) !stream.resource<external>{%size}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

} // module

// -----

// CHECK-LABEL: @no_topology_no_change
// CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !hal.buffer_view)
util.func public @no_topology_no_change(%buffer_view: !hal.buffer_view) -> !stream.resource<external> {
  %size = arith.constant 4 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %buffer_view : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%size}

  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[IMPORT]]
  %transfer = stream.async.transfer %imported : !stream.resource<external>{%size} -> !stream.resource<external>{%size}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}
