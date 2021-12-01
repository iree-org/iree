// RUN: iree-opt -split-input-file -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @FoldTimepointExport
func @FoldTimepointExport(%arg0: !hal.semaphore, %arg1: index) -> (!hal.semaphore, index) {
  // CHECK-NOT: stream.timepoint.import
  %0 = stream.timepoint.import %arg0, %arg1 : (!hal.semaphore, index) => !stream.timepoint
  // CHECK-NOT: stream.timepoint.export
  %1:2 = stream.timepoint.export %0 => (!hal.semaphore, index)
  // CHECK: return %arg0, %arg1
  return %1#0, %1#1 : !hal.semaphore, index
}

// -----

// CHECK-LABEL: @DontFoldTimepointExportMismatch
func @DontFoldTimepointExportMismatch(%arg0: !hal.semaphore, %arg1: index) -> (!hal.semaphore, i32) {
  // CHECK: stream.timepoint.import
  %0 = stream.timepoint.import %arg0, %arg1 : (!hal.semaphore, index) => !stream.timepoint
  // CHECK-NEXT: stream.timepoint.export
  %1:2 = stream.timepoint.export %0 => (!hal.semaphore, i32)
  return %1#0, %1#1 : !hal.semaphore, i32
}

// -----

// CHECK-LABEL: @FoldTimepointJoinOp
func @FoldTimepointJoinOp(%arg0: !stream.timepoint) -> !stream.timepoint {
  // CHECK-NOT: stream.timepoint.join
  %0 = stream.timepoint.join max(%arg0) => !stream.timepoint
  // CHECK: return %arg0
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateTimepointJoinOperands
func @ElideImmediateTimepointJoinOperands(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> !stream.timepoint {
  %0 = stream.timepoint.immediate => !stream.timepoint
  %1 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: = stream.timepoint.join max(%arg0, %arg1)
  %2 = stream.timepoint.join max(%arg0, %0, %1, %arg1) => !stream.timepoint
  return %2 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateTimepointJoinOperandsAll
func @ElideImmediateTimepointJoinOperandsAll() -> !stream.timepoint {
  %0 = stream.timepoint.immediate => !stream.timepoint
  %1 = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: stream.timepoint.join
  %2 = stream.timepoint.join max(%0, %1) => !stream.timepoint
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  // CHECK: return %[[IMM]]
  return %2 : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldDuplicateTimepointJoinOperands
func @FoldDuplicateTimepointJoinOperands(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: = stream.timepoint.join max(%arg0, %arg1)
  %0 = stream.timepoint.join max(%arg0, %arg1, %arg0, %arg1) => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ExpandTimepointJoinOperands
func @ExpandTimepointJoinOperands(%arg0: !stream.timepoint, %arg1: !stream.timepoint, %arg2: !stream.timepoint, %arg3: !stream.timepoint) -> !stream.timepoint {
  %join0 = stream.timepoint.join max(%arg0, %arg1) => !stream.timepoint
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%arg2, %arg0, %arg1, %arg3)
  %join1 = stream.timepoint.join max(%arg2, %join0, %arg3) => !stream.timepoint
  // CHECK: return %[[JOIN]]
  return %join1 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateAwaits
func @ElideImmediateAwaits(%arg0: !stream.resource<staging>) -> !stream.resource<staging> {
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.timepoint.immediate
  %0 = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: stream.timepoint.await
  %1 = stream.timepoint.await %0 => %arg0 : !stream.resource<staging>{%c100}
  // CHECK: return %arg0
  return %1 : !stream.resource<staging>
}

// -----

// Ensures that the await moves to the first common dominator of bb2/bb3 that
// use the awaited resources.

// CHECK-LABEL: @SinkAwaitToFirstConsumer
func @SinkAwaitToFirstConsumer(
  %arg0: i1, %arg1: i1,
  %arg2: !stream.resource<constant>,
  %arg3: !stream.resource<staging>,
  %arg4: !stream.resource<external>,
  %arg5: !stream.timepoint
) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-NOT: stream.timepoint.await
  %0:2 = stream.timepoint.await %arg5 => %arg2, %arg3 : !stream.resource<constant>{%c100}, !stream.resource<staging>{%c200}
  // CHECK: cond_br %arg0, ^bb1, ^bb4
  cond_br %arg0, ^bb1, ^bb4(%arg4 : !stream.resource<external>)
// CHECK: ^bb1:
^bb1:
  // CHECK: %[[READY:.+]]:2 = stream.timepoint.await %arg5 => %arg2, %arg3 : !stream.resource<constant>{%c100}, !stream.resource<staging>{%c200}
  // CHECK-NEXT: cond_br %arg1, ^bb2, ^bb3
  cond_br %arg1, ^bb2, ^bb3
// CHECK: ^bb2:
^bb2:
  // CHECK: = stream.async.transfer %[[READY]]#0
  %1 = stream.async.transfer %0#0 : !stream.resource<constant>{%c100} -> !stream.resource<external>{%c100}
  br ^bb4(%1 : !stream.resource<external>)
// CHECK: ^bb3:
^bb3:
  // CHECK: = stream.async.transfer %[[READY]]#1
  %2 = stream.async.transfer %0#1 : !stream.resource<staging>{%c200} -> !stream.resource<external>{%c200}
  br ^bb4(%2 : !stream.resource<external>)
// CHECK: ^bb4(
^bb4(%arg6: !stream.resource<external>):
  return %arg6 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @SinkSubviewsAcrossAwaits
func @SinkSubviewsAcrossAwaits(
  %arg0: !stream.resource<*>, %arg1: index,
  %arg2: !stream.timepoint
) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK: %[[READY:.+]] = stream.timepoint.await %arg2 => %arg0 : !stream.resource<*>{%arg1}
  // CHECK: %[[RET:.+]] = stream.resource.subview %[[READY]][%c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c256}
  %0 = stream.resource.subview %arg0[%c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c256}
  %1 = stream.timepoint.await %arg2 => %0 : !stream.resource<*>{%c256}
  // CHECK: return %[[RET]]
  return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @GroupAwaitsByTimepoint
func @GroupAwaitsByTimepoint(
  %arg0: !stream.timepoint,
  %arg1: !stream.resource<*>,
  %arg2: !stream.resource<*>,
  %arg3: !stream.resource<*>,
  %arg4: !stream.resource<*>
) -> (!stream.resource<*>, !stream.resource<*>, !stream.resource<*>, !stream.resource<*>) {
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c102 = arith.constant 102 : index
  %c103 = arith.constant 103 : index
  // CHECK: %[[RET:.+]]:4 = stream.timepoint.await %arg0 => %arg1, %arg2, %arg3, %arg4 :
  // CHECK-SAME: !stream.resource<*>{%c100}, !stream.resource<*>{%c101}, !stream.resource<*>{%c102}, !stream.resource<*>{%c103}
  %0 = stream.timepoint.await %arg0 => %arg1 : !stream.resource<*>{%c100}
  %1 = stream.timepoint.await %arg0 => %arg2 : !stream.resource<*>{%c101}
  %2:2 = stream.timepoint.await %arg0 => %arg3, %arg4 : !stream.resource<*>{%c102}, !stream.resource<*>{%c103}
  // CHECK-NEXT: return %[[RET]]#0, %[[RET]]#1, %[[RET]]#2, %[[RET]]#3
  return %0, %1, %2#0, %2#1 : !stream.resource<*>, !stream.resource<*>, !stream.resource<*>, !stream.resource<*>
}

// -----

// CHECK-LABEL: @FoldDuplicateAwaitResources
func @FoldDuplicateAwaitResources(
  %arg0: !stream.timepoint,
  %arg1: !stream.resource<staging>, %arg2: !stream.resource<*>
) -> (!stream.resource<staging>, !stream.resource<*>, !stream.resource<staging>, !stream.resource<staging>) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[RET:.+]]:2 = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  %0:4 = stream.timepoint.await %arg0 => %arg1, %arg2, %arg1, %arg1 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}, !stream.resource<staging>{%c100}, !stream.resource<staging>{%c100}
  // CHECK: return %[[RET]]#0, %[[RET]]#1, %[[RET]]#0, %[[RET]]#0
  return %0#0, %0#1, %0#2, %0#3 : !stream.resource<staging>, !stream.resource<*>, !stream.resource<staging>, !stream.resource<staging>
}

// -----

// CHECK-LABEL: @ElideUnusedTimepointAwaitOp
func @ElideUnusedTimepointAwaitOp(
  %arg0: !stream.timepoint,
  %arg1: !stream.resource<staging>, %arg2: !stream.resource<*>
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-NOT: stream.timepoint.await
  %0:2 = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  return
}
