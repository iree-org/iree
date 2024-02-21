// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies %s | FileCheck %s

// Tests that a normal clone-on-multiple-uses pattern has the last clone elided.
// This is what the --iree-stream-materialize-copy-on-write pass generates and
// expects us to clean up.

// CHECK-LABEL: @multiUseTiedOperand
util.func public @multiUseTiedOperand(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %c789_i32 = arith.constant 789 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[CLONE0:.+]] = stream.async.clone %[[SPLAT]]
  %clone0 = stream.async.clone %splat : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL0:.+]] = stream.async.fill %c456_i32, %[[CLONE0]]
  %fill0 = stream.async.fill %c456_i32, %clone0[%c0 to %c128 for %c128] : i32 -> %1 as !stream.resource<*>{%size}
  // CHECK-NOT: stream.async.clone
  %clone1 = stream.async.clone %splat : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL1:.+]] = stream.async.fill %c789_i32, %[[SPLAT]]
  %fill1 = stream.async.fill %c789_i32, %clone1[%c128 to %c256 for %c128] : i32 -> %3 as !stream.resource<*>{%size}
  // CHECK: util.return %[[FILL0]], %[[FILL1]]
  util.return %fill0, %fill1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests a copy of a by-value function argument gets elided.
// Since the caller passes in the last live reference the callee is allowed to
// mutate the memory in-place.

// CHECK-LABEL: @argMoveCallee
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>
util.func private @argMoveCallee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %arg : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[ARG0]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c128 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<*>
}
// CHECK: @argMoveCaller
util.func public @argMoveCaller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @argMoveCallee(%splat, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests a copy we cannot elide because the function argument is used after the
// call and passed by const-reference.

// CHECK-LABEL: @argCopyCallee
util.func private @argCopyCallee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.clone
  %clone = stream.async.clone %arg : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: stream.async.fill
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c128 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  util.return %fill : !stream.resource<*>
}
// CHECK: @argCopyCaller
util.func public @argCopyCaller(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @argCopyCallee(%splat, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %splat, %result : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that block arguments that are chained as last-use will get their
// clones elided while those that are used multiple times will not.
// The first splat is analyzed to be threaded through as the last possible
// use each time meaning that it can be mutated in place. The second splat
// is conditionally chosen to be the initial splat or the new value and as such
// needs to preserve the copy so the original splat is not mutated.

// CHECK-LABEL: @blockArgMove
// CHECK-SAME: (%[[COND:.+]]: i1
util.func private @blockArgMove(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[SPLAT0:.+]] = stream.async.splat %c123
  %splat0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[SPLAT1:.+]] = stream.async.splat %c456
  %splat1 = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: cf.br ^bb1(%[[SPLAT0]], %[[SPLAT1]]
  cf.br ^bb1(%splat0, %splat1 : !stream.resource<*>, !stream.resource<*>)
// CHECK: ^bb1(%[[BB1_ARG0:.+]]: !stream.resource<*>, %[[BB1_ARG1:.+]]: !stream.resource<*>)
^bb1(%bb1_0: !stream.resource<*>, %bb1_1: !stream.resource<*>):
  // CHECK-NOT: stream.async.clone
  %clone0 = stream.async.clone %bb1_0 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL0:.+]] = stream.async.fill %c123_i32, %[[BB1_ARG0]]
  %fill0 = stream.async.fill %c123_i32, %clone0[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[CLONE1:.+]] = stream.async.clone %[[BB1_ARG1]]
  %clone1 = stream.async.clone %bb1_1 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL1:.+]] = stream.async.fill %c456_i32, %[[CLONE1]]
  %fill1 = stream.async.fill %c456_i32, %clone1[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[SELECT:.+]] = arith.select %[[COND]], %[[SPLAT1]], %[[FILL1]]
  %bb1_1_new = arith.select %cond, %splat1, %fill1 : !stream.resource<*>
  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[FILL0]], %[[SELECT]]
  // CHECK-SAME:               ^bb2(%[[FILL0]], %[[SELECT]]
  cf.cond_br %cond, ^bb1(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>),
                 ^bb2(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>)
^bb2(%bb2_0: !stream.resource<*>, %bb2_1: !stream.resource<*>):
  util.return %bb2_0, %bb2_1 : !stream.resource<*>, !stream.resource<*>
}
