// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies %s | FileCheck %s

// Topology selects reusable storage for unannotated cross-device transfers.

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

// Transparent access also selects reuse for unannotated transfers.

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

// An inferred source affinity also permits topology-selected reuse.

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

// Explicit reuse stays attached to the transfer. Explicit copy follows the
// same pre-PR elision path as the default copy policy.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true})
  ]>
} {

// CHECK-LABEL: @elide_explicit_copy_transfer
// CHECK-SAME: (%[[RESOURCE:[^:]+]]: !stream.resource<*>)
util.func public @elide_explicit_copy_transfer(%resource: !stream.resource<*>) -> !stream.resource<*> {
  %size = arith.constant 4 : index
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%size} storage = copy
  // CHECK: util.return %[[RESOURCE]]
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

// Explicit copy and the default copy policy retain same-device elision.

// CHECK-LABEL: @elide_same_device_copy_policies
// CHECK-SAME: (%[[SOURCE:[^:]+]]: !stream.resource<*>)
util.func public @elide_same_device_copy_policies(%source: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) {
  %size = arith.constant 4 : index
  // CHECK-NOT: stream.async.transfer
  %default = stream.async.transfer %source : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_a>) !stream.resource<*>{%size}
  %copy = stream.async.transfer %source : !stream.resource<*>{%size}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_a>) !stream.resource<*>{%size} storage = copy
  // CHECK: util.return %[[SOURCE]], %[[SOURCE]]
  util.return %default, %copy : !stream.resource<*>, !stream.resource<*>
}

// -----

// A retained reusable transfer has a separate SSA result but may share source
// storage. The clone protects the original from the partial write.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true})
  ]>
} {

// CHECK-LABEL: @clone_after_topology_reuse
// CHECK-SAME: (%[[SOURCE:[^:]+]]: !stream.resource<*>)
util.func public @clone_after_topology_reuse(%source: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %value = arith.constant 123 : i32
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %source : !stream.resource<*>{%c4}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c4}
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[TRANSFER]]
  %clone = stream.async.clone %transfer : !stream.resource<*>{%c4} -> !stream.resource<*>{%c4}
  // CHECK: stream.async.fill {{.*}}, %[[CLONE]]
  %changed = stream.async.fill %value, %clone[%c0 to %c2 for %c2] : i32 -> %clone as !stream.resource<*>{%c4}
  // CHECK: util.return %[[SOURCE]], %{{.+}}
  util.return %source, %changed : !stream.resource<*>, !stream.resource<*>
}

// CHECK-LABEL: @clone_after_explicit_reuse
util.func public @clone_after_explicit_reuse(%source: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %value = arith.constant 123 : i32
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %source : !stream.resource<*>{%c4}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c4} storage = reuse
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[TRANSFER]]
  %clone = stream.async.clone %transfer : !stream.resource<*>{%c4} -> !stream.resource<*>{%c4}
  // CHECK: stream.async.fill {{.*}}, %[[CLONE]]
  %changed = stream.async.fill %value, %clone[%c0 to %c2 for %c2] : i32 -> %clone as !stream.resource<*>{%c4}
  util.return %source, %changed : !stream.resource<*>, !stream.resource<*>
}

// CHECK-LABEL: @clone_after_tied_cast
util.func public @clone_after_tied_cast(%source: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %value = arith.constant 123 : i32
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %source : !stream.resource<*>{%c4}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c4} storage = reuse
  // CHECK: %[[CAST:.+]] = stream.async.cast %[[TRANSFER]]
  %cast = stream.async.cast %transfer : !stream.resource<*>{%c4} -> !stream.resource<external>{%c4}
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[CAST]]
  %clone = stream.async.clone %cast : !stream.resource<external>{%c4} -> !stream.resource<external>{%c4}
  // CHECK: stream.async.fill {{.*}}, %[[CLONE]]
  %changed = stream.async.fill %value, %clone[%c0 to %c2 for %c2] : i32 -> %clone as !stream.resource<external>{%c4}
  util.return %source, %changed : !stream.resource<*>, !stream.resource<external>
}

util.func private @forward_reused_resource(%resource: !stream.resource<*>) -> !stream.resource<*> {
  util.return %resource : !stream.resource<*>
}

// CHECK-LABEL: @clone_after_private_forwarding
util.func public @clone_after_private_forwarding(%source: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %value = arith.constant 123 : i32
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer
  // CHECK-SAME: storage = reuse
  %transfer = stream.async.transfer %source : !stream.resource<*>{%c4}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c4} storage = reuse
  // CHECK: %[[FORWARDED:.+]] = util.call @forward_reused_resource(%[[TRANSFER]])
  %forwarded = util.call @forward_reused_resource(%transfer) : (!stream.resource<*>) -> !stream.resource<*>
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[FORWARDED]]
  %clone = stream.async.clone %forwarded : !stream.resource<*>{%c4} -> !stream.resource<*>{%c4}
  // CHECK: stream.async.fill {{.*}}, %[[CLONE]]
  %changed = stream.async.fill %value, %clone[%c0 to %c2 for %c2] : i32 -> %clone as !stream.resource<*>{%c4}
  util.return %source, %changed : !stream.resource<*>, !stream.resource<*>
}

} // module

// -----

// A public argument has unknown defining provenance after private forwarding.
// The incomplete walk must conservatively keep a mutating clone.

util.func private @forward_unknown_resource(%resource: !stream.resource<*>) -> !stream.resource<*> {
  util.return %resource : !stream.resource<*>
}

// CHECK-LABEL: @clone_after_incomplete_provenance
util.func public @clone_after_incomplete_provenance(%source: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %value = arith.constant 123 : i32
  // CHECK: %[[FORWARDED:.+]] = util.call @forward_unknown_resource
  %forwarded = util.call @forward_unknown_resource(%source) : (!stream.resource<*>) -> !stream.resource<*>
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[FORWARDED]]
  %clone = stream.async.clone %forwarded : !stream.resource<*>{%c4} -> !stream.resource<*>{%c4}
  // CHECK: stream.async.fill {{.*}}, %[[CLONE]]
  %changed = stream.async.fill %value, %clone[%c0 to %c2 for %c2] : i32 -> %clone as !stream.resource<*>{%c4}
  util.return %changed : !stream.resource<*>
}

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

// An inferred source affinity matching the target also permits elision.

// CHECK-LABEL: @inferred_same_affinity_elided
util.func public @inferred_same_affinity_elided(%buffer: !util.buffer) -> !stream.resource<external> {
  %size = arith.constant 4 : index
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import
  %imported = stream.tensor.import on(#hal.device.promise<@dev_b>) %buffer : !util.buffer -> tensor<1xi32> in !stream.resource<external>{%size}
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %imported : !stream.resource<external>{%size} ->
      to(#hal.device.promise<@dev_b>) !stream.resource<external>{%size}
  // CHECK: util.return %[[IMPORTED]]
  util.return %transfer : !stream.resource<external>
}

// -----

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

// Both unified-memory links in an unannotated chain select reusable storage.

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

// Mixed topology selects reuse for the unified link and retains the other.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_b = {unified_memory = true}),
    (@dev_b -> @dev_c = {})
  ]>
} {

// CHECK-LABEL: @mixed_topology_transfer_chain
// CHECK-SAME: (%[[RESOURCE:[^:]+]]: !stream.resource<*>)
util.func public @mixed_topology_transfer_chain(%arg0: !stream.resource<*>) -> !stream.resource<*> {
  %c1024 = arith.constant 1024 : index
  // CHECK: %[[TRANSFER_AB:.+]] = stream.async.transfer
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>)
  // CHECK-SAME: storage = reuse
  %transfer_ab = stream.async.transfer %arg0 : !stream.resource<*>{%c1024}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%c1024}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[TRANSFER_AB]]
  // CHECK-SAME: from(#hal.device.promise<@dev_b>) -> to(#hal.device.promise<@dev_c>)
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
