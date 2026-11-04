// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies %s | FileCheck %s

// Tests that transfers between unified links reuse storage.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

// CHECK-LABEL: @reuse_unified_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>)
util.func public @reuse_unified_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %size = arith.constant 4 : index
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%size}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

} // module

// -----

// Transparent access also allows cross-device storage reuse.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {transparent_access = true})
  ]>
} {

// CHECK-LABEL: @reuse_transparent_access_transfer
util.func public @reuse_transparent_access_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %size = arith.constant 4 : index
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%size}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

} // module

// -----

// Tests that transfers without explicit sources reuse storage.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

// CHECK-LABEL: @elide_implicit_source_transfer
// CHECK-SAME: (%{{.+}}: !util.buffer)
util.func public @elide_implicit_source_transfer(%buffer_view: !util.buffer) -> !stream.resource<external> {
  %size = arith.constant 4 : index
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %buffer_view : !util.buffer -> tensor<1xi32> in !stream.resource<external>{%size}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[IMPORTED]]
  // CHECK-SAME: to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %imported : !stream.resource<external>{%size} ->
      to(#hal.device.promise<@dev_b>) !stream.resource<external>{%size}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

} // module

// -----

// Explicit storage policies are preserved even on unified links.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true})
  ]>
} {

// CHECK-LABEL: @preserve_explicit_copy_transfer
util.func public @preserve_explicit_copy_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %size = arith.constant 4 : index
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = copy
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%size} storage = copy
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

// CHECK-LABEL: @preserve_explicit_reuse_transfer
util.func public @preserve_explicit_reuse_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %size = arith.constant 4 : index
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%size} storage = reuse
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

} // module

// -----

// Tests that transfers between non-unified links are not elided.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_c = {})
  ]>
} {

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

// Tests that transfers that change lifetime are not elided even with unified
// memory. Even though @dev_a and @dev_b have unified memory, the lifetime
// change from transient to external requires the transfer to be preserved.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

// CHECK-LABEL: @preserve_lifetime_changing_transfer
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>)
util.func public @preserve_lifetime_changing_transfer(%resource: !stream.resource<transient>) -> !stream.resource<external> {
  %size = arith.constant 4 : index
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESOURCE]]
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>)
  %transfer = stream.async.transfer %resource : !stream.resource<transient>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<external>{%size}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

} // module

// -----

// Tests that same-type transfers without topology info are elided
// (same-affinity optimization).

// CHECK-LABEL: @no_topology_same_type_elided
// CHECK-SAME: (%{{.+}}: !util.buffer)
util.func public @no_topology_same_type_elided(%buffer_view: !util.buffer) -> !stream.resource<external> {
  %size = arith.constant 4 : index
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %buffer_view : !util.buffer -> tensor<1xi32> in !stream.resource<external>{%size}
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %imported : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  // CHECK: util.return %[[IMPORTED]]
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests that transfer chains where both links have unified memory reuse storage
// in both transfers.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_c = {unified_memory = true}),
    (@dev_a -> @dev_c = {})
  ]>
} {

// CHECK-LABEL: @reuse_transfer_chain
// CHECK-SAME: (%[[RESOURCE:[^:]+]]: !stream.resource<*>)
util.func public @reuse_transfer_chain(%arg0: !stream.resource<*>) -> !stream.resource<*> {
  %c1024 = arith.constant 1024 : index
  // CHECK: %[[TRANSFER_AB:.+]] = stream.async.transfer %[[RESOURCE]]
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = reuse
  %transfer_ab = stream.async.transfer %arg0 : !stream.resource<*>{%c1024}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c1024}
  // CHECK: %[[TRANSFER_BC:.+]] = stream.async.transfer %[[TRANSFER_AB]]
  // CHECK-SAME: from(#hal.device.promise<@dev_b>) -> to(#hal.device.promise<@dev_c>)
  // CHECK-SAME: storage = reuse
  %transfer_bc = stream.async.transfer %transfer_ab : !stream.resource<*>{%c1024}
      from(#hal.device.promise<@dev_b>) -> to(#hal.device.promise<@dev_c>) !stream.resource<*>{%c1024}
  // CHECK: util.return %[[TRANSFER_BC]]
  util.return %transfer_bc : !stream.resource<*>
}

} // module

// -----

// Tests transfer chains with mixed topology: first reuses storage, second
// keeps its default copy policy.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_c = {})
  ]>
} {

// CHECK-LABEL: @mixed_topology_transfer_chain
util.func public @mixed_topology_transfer_chain(%arg0: !stream.resource<*>) -> !stream.resource<*> {
  %c1024 = arith.constant 1024 : index
  // CHECK: %[[TRANSFER_AB:.+]] = stream.async.transfer
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = reuse
  %transfer_ab = stream.async.transfer %arg0 : !stream.resource<*>{%c1024}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c1024}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[TRANSFER_AB]]
  // CHECK-SAME: from(#hal.device.promise<@dev_b>) -> to(#hal.device.promise<@dev_c>)
  // CHECK-NOT: storage = reuse
  %transfer_bc = stream.async.transfer %transfer_ab : !stream.resource<*>{%c1024}
      from(#hal.device.promise<@dev_b>) -> to(#hal.device.promise<@dev_c>) !stream.resource<*>{%c1024}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer_bc : !stream.resource<*>
}

} // module

// -----

// Tests same-device transfers with unified topology are elided.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

// CHECK-LABEL: @elide_same_device_transfer
// CHECK-SAME: (%[[RESOURCE:[^:]+]]: !stream.resource<*>)
util.func public @elide_same_device_transfer(%arg0: !stream.resource<*>) -> !stream.resource<*> {
  %c1024 = arith.constant 1024 : index
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %arg0 : !stream.resource<*>{%c1024}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_a>) !stream.resource<*>{%c1024}
  // CHECK: util.return %[[RESOURCE]]
  util.return %transfer : !stream.resource<*>
}

} // module

// -----

// Tests that transfers reuse storage when source affinity must be inferred
// through control flow operations that don't implement AffinityOpInterface.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_a = {unified_memory = true})
  ]>
} {

// CHECK-LABEL: @reuse_transfer_through_scf_if
// CHECK-SAME: (%[[COND:.+]]: i1, %{{.+}}: !util.buffer)
util.func public @reuse_transfer_through_scf_if(%cond: i1, %buffer: !util.buffer) -> !stream.resource<external> {
  %size = arith.constant 4 : index
  // Import on device A.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import
  %imported = stream.tensor.import on(#hal.device.promise<@dev_a>) %buffer : !util.buffer -> tensor<1xi32> in !stream.resource<external>{%size}

  // Pass through scf.if which doesn't implement AffinityOpInterface.
  // The scf.if result's affinity must be inferred by walking to stream.tensor.import.
  // CHECK: %[[RESULT:.+]] = scf.if %[[COND]]
  %result = scf.if %cond -> !stream.resource<external> {
    // CHECK: scf.yield %[[IMPORTED]]
    scf.yield %imported : !stream.resource<external>
  } else {
    // CHECK: scf.yield %[[IMPORTED]]
    scf.yield %imported : !stream.resource<external>
  }

  // Transfer to device B without explicit source affinity.
  // tryInferValueAffinity should walk through the scf.if to find the affinity.
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[RESULT]]
  // CHECK-SAME: to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %result : !stream.resource<external>{%size} ->
      to(#hal.device.promise<@dev_b>) !stream.resource<external>{%size}

  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

} // module
