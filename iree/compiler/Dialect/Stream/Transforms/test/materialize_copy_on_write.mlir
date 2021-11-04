// RUN: iree-opt -split-input-file -pass-pipeline='builtin.func(iree-stream-materialize-copy-on-write)' %s | IreeFileCheck %s

// Tests that block arguments (including function arguments) are always cloned.
// Until a whole-program analysis runs we don't know their semantics.

// CHECK-LABEL: @blockArgsNeedCopies
//  CHECK-SAME: (%[[SRC:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
func @blockArgsNeedCopies(%src: !stream.resource<*>, %size: index) -> !stream.resource<*> {
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
func @singleUseTiedOperand(%size: index) -> !stream.resource<*> {
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
// copy will be elided with the -iree-stream-elide-async-copies pass.

// CHECK-LABEL: @multiUseTiedOperand
//  CHECK-SAME: (%[[SIZE:.+]]: index)
func @multiUseTiedOperand(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
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

// Tests that block args (like function args) are copied until copy elision can
// take care of them later.

// CHECK-LABEL: @blockArgMove
func @blockArgMove(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %splat0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %splat1 = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%size}
  br ^bb1(%splat0, %splat1 : !stream.resource<*>, !stream.resource<*>)
// CHECK: ^bb1(%[[ARG0:.+]]: !stream.resource<*>, %[[ARG1:.+]]: !stream.resource<*>)
^bb1(%bb1_0: !stream.resource<*>, %bb1_1: !stream.resource<*>):
  // CHECK: %[[CLONE0:.+]] = stream.async.clone %[[ARG0]]
  // CHECK: stream.async.fill %c123_i32, %[[CLONE0]]
  %fill0 = stream.async.fill %c123_i32, %bb1_0[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[CLONE1:.+]] = stream.async.clone %[[ARG1]]
  // CHECK: stream.async.fill %c456_i32, %[[CLONE1]]
  %fill1 = stream.async.fill %c456_i32, %bb1_1[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  %bb1_1_new = select %cond, %splat1, %fill1 : !stream.resource<*>
  cond_br %cond, ^bb1(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>),
                 ^bb2(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>)
^bb2(%bb2_0: !stream.resource<*>, %bb2_1: !stream.resource<*>):
  return %bb2_0, %bb2_1 : !stream.resource<*>, !stream.resource<*>
}
