// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-stream-materialize-copy-on-write))' %s | FileCheck %s

// Tests that block arguments (including function arguments) are always cloned.
// Until a whole-program analysis runs we don't know their semantics.

// CHECK-LABEL: @blockArgsNeedCopies
//  CHECK-SAME: (%[[SRC:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
func.func @blockArgsNeedCopies(%src: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SRC]] : !stream.resource<*>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[CLONE]]{{.+}} -> %[[CLONE]]
  %0 = stream.async.fill %c123_i32, %src[%c0 to %c128 for %c128] : i32 -> %src as !stream.resource<*>{%size}
  // CHECK: return %[[FILL]]
  return %0 : !stream.resource<*>
}

// -----

// Tests that copies are not inserted where they are trivially not needed.

// CHECK-LABEL: @singleUseTiedOperand
//  CHECK-SAME: (%[[SIZE:.+]]: index)
func.func @singleUseTiedOperand(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %c789_i32 = arith.constant 789 : i32
  // CHECK: stream.async.splat
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK-NOT: stream.async.clone
  // CHECK: stream.async.fill
  %1 = stream.async.fill %c456_i32, %0[%c0 to %c128 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  // CHECK-NOT: stream.async.clone
  // CHECK: stream.async.fill
  %2 = stream.async.fill %c789_i32, %1[%c128 to %c256 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  return %2 : !stream.resource<*>
}

// -----

// Tests that copies are inserted when there are multiple uses of a mutated
// value (in this case, the splat acting as an initializer). The additional
// copy will be elided with the --iree-stream-elide-async-copies pass.

// CHECK-LABEL: @multiUseTiedOperand
//  CHECK-SAME: (%[[SIZE:.+]]: index)
func.func @multiUseTiedOperand(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %c789_i32 = arith.constant 789 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[CLONE0:.+]] = stream.async.clone %[[SPLAT]]
  // CHECK: %[[FILL0:.+]] = stream.async.fill %c456_i32, %[[CLONE0]]
  %1 = stream.async.fill %c456_i32, %0[%c0 to %c128 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  // CHECK: %[[CLONE1:.+]] = stream.async.clone %[[SPLAT]]
  // CHECK: %[[FILL1:.+]] = stream.async.fill %c789_i32, %[[CLONE1]]
  %2 = stream.async.fill %c789_i32, %0[%c128 to %c256 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  return %1, %2 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests collectives with naturally tied results.
// TODO(#11249): support in-place collectives - when supported this will become
// a negative test as we'd expect %send_recv to be used for both operands.

// CHECK-LABEL: @tiedCollectivesTODO
//  CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel, %[[SEND_RECV:.+]]: !stream.resource<*>, %[[SEND_SIZE:.+]]: index, %[[RECV_SIZE:.+]]: index, %[[COUNT:.+]]: index)
func.func private @tiedCollectivesTODO(%channel: !stream.channel, %send_recv: !stream.resource<*>, %send_size: index, %recv_size: index, %count: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RECV_CLONE:.+]] = stream.async.clone on(#hal.affinity.queue<[0]>) %[[SEND_RECV]]
  // CHECK: %[[ALL_GATHER:.+]] = stream.async.collective<all_gather : f32>[%[[COUNT]]]
  %0 = stream.async.collective<all_gather : f32>[%count] on(#hal.affinity.queue<[0]>) channel(%channel)
      // CHECK-SAME: %[[SEND_RECV]][%c0 to %[[SEND_SIZE]] for %[[SEND_SIZE]]],
      %send_recv[%c0 to %send_size for %send_size],
      // CHECK-SAME: %[[RECV_CLONE]][%c0 to %[[RECV_SIZE]] for %[[RECV_SIZE]]] :
      %send_recv[%c0 to %recv_size for %recv_size] :
      // CHECK-SAME: !stream.resource<*>{%[[SEND_SIZE]]} -> %[[RECV_CLONE]] as !stream.resource<*>{%[[RECV_SIZE]]}
      !stream.resource<*>{%send_size} -> %recv as !stream.resource<*>{%recv_size}
  // CHECK: return %[[ALL_GATHER]]
  return %0 : !stream.resource<*>
}

// -----

// Tests tied dispatches with a data dependency.
// %splat0 is mutated by @dispatch0 and a clone gets inserted to preserve its
// original contents for use by @dispatch1.

// CHECK-LABEL: @tiedDispatches
func.func private @tiedDispatches() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c40 = arith.constant 40 : index

  // CHECK: %[[SPLAT0:.+]] = stream.async.splat %c0_i32
  %splat0 = stream.async.splat %c0_i32 : i32 -> !stream.resource<*>{%c40}
  // CHECK: %[[SPLAT1:.+]] = stream.async.splat %c1_i32
  %splat1 = stream.async.splat %c1_i32 : i32 -> !stream.resource<*>{%c16}

  // CHECK: %[[CLONE0:.+]] = stream.async.clone %[[SPLAT0]]
  // CHECK: %[[DISPATCH0:.+]] = stream.async.dispatch @ex::@dispatch0[%c1, %c1, %c1](%[[CLONE0]][%c0 to %c40 for %c40], %[[SPLAT1]][%c0 to %c16 for %c16])
  // CHECK-SAME: (!stream.resource<*>{%c40}, !stream.resource<*>{%c16}) -> %[[CLONE0]]{%c40}
  %dispatch0 = stream.async.dispatch @ex::@dispatch0[%c1, %c1, %c1](%splat0[%c0 to %c40 for %c40], %splat1[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c40}, !stream.resource<*>{%c16}) -> %splat0{%c40}

  // CHECK: %[[DISPATCH1:.+]] = stream.async.dispatch @ex::@dispatch1[%c1, %c1, %c1](%[[DISPATCH0]][%c0 to %c40 for %c40], %[[SPLAT0]][%c0 to %c40 for %c40])
  // CHECK-SAME: (!stream.resource<*>{%c40}, !stream.resource<*>{%c40}) -> %[[DISPATCH0]]{%c40}
  %dispatch1 = stream.async.dispatch @ex::@dispatch1[%c1, %c1, %c1](%dispatch0[%c0 to %c40 for %c40], %splat0[%c0 to %c40 for %c40]) : (!stream.resource<*>{%c40}, !stream.resource<*>{%c40}) -> %dispatch0{%c40}

  return
}

// -----

// Tests that block args (like function args) are copied until copy elision can
// take care of them later.

// CHECK-LABEL: @blockArgMove
func.func @blockArgMove(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %splat0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %splat1 = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%size}
  cf.br ^bb1(%splat0, %splat1 : !stream.resource<*>, !stream.resource<*>)
// CHECK: ^bb1(%[[ARG0:.+]]: !stream.resource<*>, %[[ARG1:.+]]: !stream.resource<*>)
^bb1(%bb1_0: !stream.resource<*>, %bb1_1: !stream.resource<*>):
  // CHECK: %[[CLONE0:.+]] = stream.async.clone %[[ARG0]]
  // CHECK: stream.async.fill %c123_i32, %[[CLONE0]]
  %fill0 = stream.async.fill %c123_i32, %bb1_0[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[CLONE1:.+]] = stream.async.clone %[[ARG1]]
  // CHECK: stream.async.fill %c456_i32, %[[CLONE1]]
  %fill1 = stream.async.fill %c456_i32, %bb1_1[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  %bb1_1_new = arith.select %cond, %splat1, %fill1 : !stream.resource<*>
  cf.cond_br %cond, ^bb1(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>),
                 ^bb2(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>)
^bb2(%bb2_0: !stream.resource<*>, %bb2_1: !stream.resource<*>):
  return %bb2_0, %bb2_1 : !stream.resource<*>, !stream.resource<*>
}
