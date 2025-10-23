// RUN: iree-opt --split-input-file --iree-stream-materialize-transient-size-queries %s | FileCheck %s

// Tests no transients (pass should be no-op).
// CHECK-LABEL: @no_transients
util.func public @no_transients(%arg0: !stream.resource<*>, %arg0_size: index) -> (!stream.resource<*>, index) {
  // CHECK-NOT: iree.reflection
  // CHECK: util.return
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests constant size transients.

// CHECK-LABEL: @constant_size
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !hal.buffer)
// CHECK-SAME: iree.abi.transients.size = @constant_size_transients_size
util.func public @constant_size(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !hal.buffer
) -> (!stream.resource<*>, index) {
  %c1024 = arith.constant 1024 : index
  // Pack should have attribute stripped after pass runs.
  %0:2 = stream.resource.pack slices({
    [0, 0] = %c1024
  }) : index attributes {stream.experimental.transients}
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// CHECK: util.func public @constant_size_transients_size
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !hal.buffer)
// CHECK: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK: %[[PACK:.+]]:2 = stream.resource.pack
// CHECK: util.return %[[PACK]]#0 : index

// -----

// Tests with dynamic size transients.

// CHECK-LABEL: @dynamic_size
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !hal.buffer)
// CHECK-SAME: iree.abi.transients.size = @dynamic_size_transients_size
util.func public @dynamic_size(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !hal.buffer
) -> (!stream.resource<*>, index) {
  %c4 = arith.constant 4 : index
  %size = arith.muli %arg0_size, %c4 : index
  // Pack with dynamic size computation.
  %0:2 = stream.resource.pack slices({
    [0, 0] = %size
  }) : index attributes {stream.experimental.transients}
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// CHECK: util.func public @dynamic_size_transients_size
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !hal.buffer)
//      CHECK: %[[C4:.+]] = arith.constant 4 : index
//      CHECK: %[[SIZE:.+]] = arith.muli %[[ARG0_SIZE]], %[[C4]] : index
//      CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[SIZE]]
// CHECK-NEXT: }) : index
//      CHECK: util.return %[[PACK]]#0 : index

// -----

// Tests with max arithmetic (from mutually exclusive branches).

// CHECK-LABEL: @max_arithmetic
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !hal.buffer)
// CHECK-SAME: iree.abi.transients.size = @max_arithmetic_transients_size
util.func public @max_arithmetic(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !hal.buffer
) -> (!stream.resource<*>, index) {
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %0 = arith.maxui %c256, %c256 : index
  %1 = arith.maxui %0, %c512 : index
  // Pack with max arithmetic from mutually exclusive branch handling.
  %2:2 = stream.resource.pack slices({
    [0, 0] = %1
  }) : index attributes {stream.experimental.transients}
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// CHECK: util.func public @max_arithmetic_transients_size
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !hal.buffer)
//      CHECK: %[[C256:.+]] = arith.constant 256 : index
//      CHECK: %[[C512:.+]] = arith.constant 512 : index
//      CHECK: %[[MAX0:.+]] = arith.maxui %[[C256]], %[[C256]] : index
//      CHECK: %[[MAX1:.+]] = arith.maxui %[[MAX0]], %[[C512]] : index
//      CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[MAX1]]
// CHECK-NEXT: }) : index
//      CHECK: util.return %[[PACK]]#0 : index

// -----

// Tests with multiple packs in one function.

// CHECK-LABEL: @multiple_packs
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !hal.buffer)
// CHECK-SAME: iree.abi.transients.size = @multiple_packs_transients_size
util.func public @multiple_packs(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !hal.buffer
) -> (!stream.resource<*>, index) {
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  // First pack.
  %0:3 = stream.resource.pack slices({
    [0, 0] = %c512,
    [0, 0] = %c1024
  }) : index attributes {stream.experimental.transients}
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// CHECK: util.func public @multiple_packs_transients_size
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !hal.buffer)
//      CHECK: %[[C512:.+]] = arith.constant 512 : index
//      CHECK: %[[C1024:.+]] = arith.constant 1024 : index
//      CHECK: %[[PACK:.+]]:3 = stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[C512]],
// CHECK-NEXT:   [0, 0] = %[[C1024]]
// CHECK-NEXT: }) : index
//      CHECK: util.return %[[PACK]]#0 : index

// -----

// Tests with execute op to ensure it's not pulled into size query.

// CHECK-LABEL: @with_execute
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[TP:[a-z0-9]+]]: !stream.timepoint)
// CHECK-SAME: iree.abi.transients.size = @with_execute_transients_size
util.func public @with_execute(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>,
  %tp: !stream.timepoint
) -> (!stream.resource<*>, index) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1024 = arith.constant 1024 : index
  // Pack with constant size.
  %0:2 = stream.resource.pack slices({
    [0, 0] = %c1024
  }) : index attributes {stream.experimental.transients}
  %1 = stream.resource.subview %storage[%0#1] : !stream.resource<transient>{%0#0} -> !stream.resource<transient>{%c1024}
  // Execute should NOT be cloned into size query.
  %2 = stream.cmd.execute await(%tp) => with(%1 as %arg3: !stream.resource<transient>{%c1024}) {
    stream.cmd.fill %c0_i32, %arg3[%c0 for %c1024] : i32 -> !stream.resource<transient>{%c1024}
  } => !stream.timepoint
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// CHECK: util.func public @with_execute_transients_size
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[TP:[a-z0-9]+]]: !stream.timepoint)
//      CHECK: %[[C1024:.+]] = arith.constant 1024 : index
//      CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[C1024]]
// CHECK-NEXT: }) : index
// Size query should only have the constant, not the execute.
//  CHECK-NOT: stream.cmd.execute
//  CHECK-NOT: stream.resource.subview
//      CHECK: util.return %[[PACK]]#0 : index
