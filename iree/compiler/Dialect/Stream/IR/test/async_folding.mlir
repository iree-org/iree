// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// Ensures that the splat moves to the first common dominator of bb2/bb3.
// We likely want to clone instead to reduce lifetime of the splats.

// CHECK-LABEL: @SinkSplatsToConsumers
func @SinkSplatsToConsumers(
  %arg0: i1, %arg1: i1,
  %arg2: !stream.resource<*>,
  %arg3: !stream.resource<*>,
  %arg4: !stream.resource<*>
) -> !stream.resource<*> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.splat
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c100}
  // CHECK: cond_br %arg0, ^bb1, ^bb4
  cond_br %arg0, ^bb1, ^bb4(%arg4 : !stream.resource<*>)
// CHECK: ^bb1:
^bb1:
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c100}
  // CHECK-NEXT: cond_br %arg1, ^bb2, ^bb3
  cond_br %arg1, ^bb2, ^bb3
// CHECK: ^bb2:
^bb2:
  // CHECK: = stream.async.dispatch @executable::@dispatch0[%c1, %c2, %c3](%[[SPLAT]])
  %2 = stream.async.dispatch @executable::@dispatch0[%c1, %c2, %c3](%0) : (!stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
  br ^bb4(%2 : !stream.resource<*>)
// CHECK: ^bb3:
^bb3:
  // CHECK: = stream.async.dispatch @executable::@dispatch1[%c1, %c2, %c3](%[[SPLAT]])
  %3 = stream.async.dispatch @executable::@dispatch1[%c1, %c2, %c3](%0) : (!stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
  br ^bb4(%3 : !stream.resource<*>)
// CHECK: ^bb4(
^bb4(%arg6: !stream.resource<*>):
  return %arg6 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @PropagateClonableOps
func @PropagateClonableOps(%arg0: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[T:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%arg0}
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%arg0}
  // CHECK-NOT: stream.async.clone
  %1 = stream.async.clone %0 : !stream.resource<*>{%arg0} -> !stream.resource<*>{%arg0}
  // CHECK: return %[[T]]
  return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @ConvertSplatConstantsIntoSplats
func @ConvertSplatConstantsIntoSplats(%arg0: index) -> (!stream.resource<transient>, !stream.resource<transient>) {
  // CHECK-NOT: = stream.async.constant : !stream.resource<transient>{%arg0} = dense<[3]> : tensor<8xi32>
  // CHECK: %[[CST:.+]] = arith.constant 3 : i32
  // CHECK: %0 = stream.async.splat %[[CST]] : i32 -> !stream.resource<transient>{%arg0}
  %0 = stream.async.constant : !stream.resource<transient>{%arg0} = dense<3> : tensor<8xi32>
  // CHECK: = stream.async.constant : !stream.resource<transient>{%arg0} = dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
  %1 = stream.async.constant : !stream.resource<transient>{%arg0} = dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
  return %0, %1 : !stream.resource<transient>, !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncSliceOp
func @FoldAsyncSliceOp(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.async.slice
  %0 = stream.async.slice %arg0[%c0 to %arg1] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
  // CHECK: return %arg0
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @PropagateSplatsThroughSlices
func @PropagateSplatsThroughSlices(%arg0: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[T:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c128}
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%arg0}
  // CHECK-NOT: stream.async.slice
  %1 = stream.async.slice %0[%c0 to %c128] : !stream.resource<*>{%arg0} -> !stream.resource<*>{%c128}
  // CHECK: return %[[T]]
  return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @FlattenFullFillToSplat
func @FlattenFullFillToSplat(%arg0: !stream.resource<*>, %arg1: index, %arg2: f32) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[T:.+]] = stream.async.splat %arg2 : f32 -> !stream.resource<*>{%arg1}
  %0 = stream.async.fill %arg2, %arg0[%c0 to %arg1 for %arg1] : f32 -> %arg0 as !stream.resource<*>{%arg1}
  // CHECK: return %[[T]]
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @FoldAsyncUpdateOp
func @FoldAsyncUpdateOp(%arg0: !stream.resource<*>, %arg1: !stream.resource<*>, %arg2: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.async.update
  %0 = stream.async.update %arg1, %arg0[%c0 to %arg2] : !stream.resource<*>{%arg2} -> %arg0 as !stream.resource<*>{%arg2}
  // CHECK: return %arg1
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @CombineSplatUpdateFromToFill
func @CombineSplatUpdateFromToFill(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.splat
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c128}
  // CHECK: %[[T:.+]] = stream.async.fill %c123_i32, %arg0[%c0 to %c128 for %c128] : i32 -> %arg0 as !stream.resource<*>{%arg1}
  %1 = stream.async.update %0, %arg0[%c0 to %c128] : !stream.resource<*>{%c128} -> %arg0 as !stream.resource<*>{%arg1}
  // CHECK: return %[[T]]
  return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @CombineSliceUpdateFromToCopy
func @CombineSliceUpdateFromToCopy(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: stream.async.slice
  %0 = stream.async.slice %arg0[%c0 to %c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
  // CHECK: %[[T:.+]] = stream.async.copy %arg0[%c0 to %c128], %arg2[%c0 to %c128], %c128 : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg3}
  %1 = stream.async.update %0, %arg2[%c0 to %c128] : !stream.resource<*>{%c128} -> %arg2 as !stream.resource<*>{%arg3}
  // CHECK: return %[[T]]
  return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @AsyncCopyFullSourceToUpdate
func @AsyncCopyFullSourceToUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // This copy is from the full source (0..%arg3) so it can be turned into an update.
  // CHECK: = stream.async.update %arg2, %arg0[%c0 to %arg3] : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.copy %arg2[%c0 to %arg3], %arg0[%c0 to %arg3], %arg3 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}

  // This copy is only a partial section of the source and needs to remain a copy.
  // CHECK: = stream.async.copy %arg2[%c16 to %arg3], %arg0[%c0 to %arg3], %c8 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  %1 = stream.async.copy %arg2[%c16 to %arg3], %arg0[%c0 to %arg3], %c8 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}

  return %0, %1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// CHECK-LABEL: @FoldAsyncTransferOp
func @FoldAsyncTransferOp(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.resource<transient> {
  // CHECK-NOT: stream.async.transfer
  %0 = stream.async.transfer %arg0 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg1}
  %1 = stream.async.transfer %0 : !stream.resource<staging>{%arg1} -> !stream.resource<transient>{%arg1}
  return %1 : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @RedundantTransferElision
func @RedundantTransferElision(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.resource<transient> {
  // CHECK-NOT: stream.async.transfer
  %0 = stream.async.transfer %arg0 : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%arg1}
  return %0 : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @ElideImmediateAsyncExecuteWaits
func @ElideImmediateAsyncExecuteWaits(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.timepoint) {
  %c1 = arith.constant 1 : index
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: stream.async.execute with
  %0:2 = stream.async.execute await(%imm) => with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> %arg0 as !stream.resource<*>{%arg1} {
    // CHECK: stream.async.dispatch
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg2) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK: stream.yield
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ChainAsyncExecuteWaits
func @ChainAsyncExecuteWaits(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  %c1 = arith.constant 1 : index
  // CHECK-NOT: stream.timepoint.await
  %0 = stream.timepoint.await %arg2 => %arg0 : !stream.resource<*>{%arg1}
  // CHECK: stream.async.execute await(%arg2) => with
  %1:2 = stream.async.execute with(%0 as %arg3: !stream.resource<*>{%arg1}) -> %0 as !stream.resource<*>{%arg1} {
    // CHECK: stream.async.dispatch
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK: stream.yield
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  return %1#0, %1#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @CloneCapturedAsyncExecuteSubviewOps
func @CloneCapturedAsyncExecuteSubviewOps(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: stream.resource.subview
  %0 = stream.resource.subview %arg0[%c0] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
  // CHECK: = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> %arg0{%c128}
  %1:2 = stream.async.execute with(%0 as %arg3: !stream.resource<*>{%c128}) -> %0{%c128} {
    // CHECK: %[[T:.+]] = stream.resource.subview %arg2[%c0] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
    // CHECK: stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%[[T]])
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK: stream.yield
    stream.yield %1 : !stream.resource<*>{%c128}
  } => !stream.timepoint
  return %1#0, %1#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideNoOpAsyncExecuteOp
func @ElideNoOpAsyncExecuteOp(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK-NOT: stream.async.execute
  %1:2 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> %arg0{%arg1} {
    stream.yield %arg3 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  // CHECK: return %arg0, %[[IMM]]
  return %1#0, %1#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @TieRegionResultsAsyncExecuteOp
func @TieRegionResultsAsyncExecuteOp(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.timepoint) {
  %c1 = arith.constant 1 : index
  // CHECK: = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> %arg0{%arg1}
  %0:2 = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
    // CHECK: %[[T:.+]] = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg2)
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg2) : (!stream.resource<*>{%arg1}) -> %arg2{%arg1}
    // CHECK: stream.yield %[[T]]
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideUnusedAsyncExecuteOp
func @ElideUnusedAsyncExecuteOp(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) {
  %c1 = arith.constant 1 : index
  // CHECK-NOT: stream.async.execute
  %0:2 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  return
}

// -----

// CHECK-LABEL: @TieRegionResultsAsyncConcurrentOp
func @TieRegionResultsAsyncConcurrentOp(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.timepoint) {
  %c1 = arith.constant 1 : index
  // CHECK: = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> %arg0{%arg1}
  %0:2 = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
    // CHECK: %[[EXEC_T:.+]] = stream.async.concurrent with(%arg2 as %arg3: !stream.resource<*>{%arg1}) -> %arg2{%arg1}
    %1 = stream.async.concurrent with(%arg2 as %arg3: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
      // CHECK: %[[WAVE_T:.+]] = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3) : (!stream.resource<*>{%arg1}) -> %arg3{%arg1}
      %2 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3) : (!stream.resource<*>{%arg1}) -> %arg3{%arg1}
      // CHECK: stream.yield %[[WAVE_T]]
      stream.yield %2 : !stream.resource<*>{%arg1}
    }
    // CHECK: stream.yield %[[EXEC_T]]
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideUnusedAsyncConcurrentOp
func @ElideUnusedAsyncConcurrentOp(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  %c1 = arith.constant 1 : index
  // CHECK: stream.async.execute
  %0:2 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
    // CHECK: stream.async.dispatch @executable::@dispatch0
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK-NOT: stream.async.concurrent
    %2 = stream.async.concurrent with(%arg3 as %arg4: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
      // CHECK-NOT: stream.async.dispatch @executable::@dispatch1
      %3 = stream.async.dispatch @executable::@dispatch1[%c1, %c1, %c1](%arg4) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
      stream.yield %3 : !stream.resource<*>{%arg1}
    }
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}
