// RUN: iree-opt --split-input-file --iree-stream-elide-async-transfers %s | FileCheck %s

// Tests that a transfer with no source/target is turned into a clone.

// CHECK-LABEL: @unassignedTransfers
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
util.func public @unassignedTransfers(%resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[TRANSFER:.+]] = stream.async.clone %[[RESOURCE]] : !stream.resource<*>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %transfer = stream.async.transfer %resource : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

// -----

// Tests that an elided transfer which changes lifetime has the proper types on
// the resulting clone op.

// CHECK-LABEL: @unassignedLifetimeTransfers
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @unassignedLifetimeTransfers(%resource: !stream.resource<transient>, %size: index) -> !stream.resource<external> {
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[TRANSFER:.+]] = stream.async.clone %[[RESOURCE]] : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %transfer = stream.async.transfer %resource : !stream.resource<transient>{%size} -> !stream.resource<external>{%size}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests that transfers to/from staging are not converted to clones.

// CHECK-LABEL: @ignoreStagingTransfers
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<staging>, %[[SIZE:.+]]: index)
util.func public @ignoreStagingTransfers(%resource: !stream.resource<staging>, %size: index) -> !stream.resource<external> {
  // CHECK: stream.async.transfer
  %transfer = stream.async.transfer %resource : !stream.resource<staging>{%size} -> !stream.resource<external>{%size}
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests that missing source affinities from the target are elided.

// CHECK-LABEL: @omittedSourceAffinity
util.func public @omittedSourceAffinity(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[TRANSFER:.+]] = stream.async.clone on(#hal.device.promise<@dev_a>) %[[SPLAT]]
  %transfer = stream.async.transfer %splat : !stream.resource<*>{%size} -> to(#hal.device.promise<@dev_a>) !stream.resource<*>{%size}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}

// -----

// Tests that missing target affinities from the source are elided.

// CHECK-LABEL: @omittedTargetAffinity
util.func public @omittedTargetAffinity(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK-NOT: stream.async.transfer
  // CHECK: %[[TRANSFER:.+]] = stream.async.clone on(#hal.device.promise<@dev_a>) %[[SPLAT]]
  %transfer = stream.async.transfer %splat : !stream.resource<*>{%size} from(#hal.device.promise<@dev_a>) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<*>
}
