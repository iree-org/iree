// RUN: iree-opt --split-input-file --canonicalize=test-convergence=true %s | iree-opt --split-input-file | FileCheck %s

// Ensures that the splat moves to the first common dominator of bb2/bb3.
// We likely want to clone instead to reduce lifetime of the splats.

// CHECK-LABEL: @SinkSplatsToConsumers
util.func private @SinkSplatsToConsumers(
  %arg0: i1, %arg1: i1,
  %arg2: !stream.resource<*>,
  %arg3: !stream.resource<*>,
  %arg4: !stream.resource<*>
) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.splat
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c100}
  // CHECK: cf.cond_br %arg0, ^bb1, ^bb4
  cf.cond_br %arg0, ^bb1, ^bb4(%arg4 : !stream.resource<*>)
// CHECK: ^bb1:
^bb1:
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c100}
  // CHECK-NEXT: cf.cond_br %arg1, ^bb2, ^bb3
  cf.cond_br %arg1, ^bb2, ^bb3
// CHECK: ^bb2:
^bb2:
  // CHECK: = stream.async.dispatch @executable::@dispatch0[%c1, %c2, %c3](%[[SPLAT]][%c0 to %c100 for %c100])
  %2 = stream.async.dispatch @executable::@dispatch0[%c1, %c2, %c3](%0[%c0 to %c100 for %c100]) : (!stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
  cf.br ^bb4(%2 : !stream.resource<*>)
// CHECK: ^bb3:
^bb3:
  // CHECK: = stream.async.dispatch @executable::@dispatch1[%c1, %c2, %c3](%[[SPLAT]][%c0 to %c100 for %c100])
  %3 = stream.async.dispatch @executable::@dispatch1[%c1, %c2, %c3](%0[%c0 to %c100 for %c100]) : (!stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
  cf.br ^bb4(%3 : !stream.resource<*>)
// CHECK: ^bb4(
^bb4(%arg6: !stream.resource<*>):
  util.return %arg6 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @SinkSplatsToCommonAncestorOfConsumersInRegions
util.func public @SinkSplatsToCommonAncestorOfConsumersInRegions(%arg0: i1) -> (!stream.resource<*>, !stream.resource<*>) {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  %c2 = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
  %c3 = arith.constant 3 : index
  // CHECK-DAG: %[[C100:.+]] = arith.constant 100 : index
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[C123:.+]] = arith.constant 123 : i32
  %c123_i32 = arith.constant 123 : i32
  // CHECK-DAG: %[[C456:.+]] = arith.constant 456 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK-DAG: %[[C789:.+]] = arith.constant 789 : i32
  %c789_i32 = arith.constant 789 : i32
  // CHECK-NOT: stream.async.splat %[[C123]]
  // CHECK-NOT: stream.async.splat %[[C456]]
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c100}
  %1 = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c100}
  // CHECK: %[[SPLAT3:.+]] = stream.async.splat %[[C789]]
  %2 = stream.async.splat %c789_i32 : i32 -> !stream.resource<*>{%c100}
  // CHECK: stream.async.dispatch @executable::@dispatch2[%[[C1]], %[[C2]], %[[C3]]](%[[SPLAT3]][%[[C0]] to %[[C100]] for %[[C100]]])
  %3 = stream.async.dispatch @executable::@dispatch2[%c1, %c2, %c3](%2[%c0 to %c100 for %c100]) : (!stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
  // CHECK-DAG: %[[SPLAT1:.+]] = stream.async.splat %[[C123]]
  // CHECK-DAG: %[[SPLAT2:.+]] = stream.async.splat %[[C456]]
  // CHECK-NEXT: scf.if
  %4 = scf.if %arg0 -> (!stream.resource<*>) {
    // CHECK: stream.async.dispatch @executable::@dispatch0[%[[C1]], %[[C2]], %[[C3]]](%[[SPLAT1]][%[[C0]] to %[[C100]] for %[[C100]]], %[[SPLAT2]][%[[C0]] to %[[C100]] for %[[C100]]])
    %5 = stream.async.dispatch @executable::@dispatch0[%c1, %c2, %c3](%0[%c0 to %c100 for %c100], %1[%c0 to %c100 for %c100]) : (!stream.resource<*>{%c100}, !stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
    scf.yield %5 : !stream.resource<*>
  // CHECK: else
  } else {
    // CHECK: stream.async.dispatch @executable::@dispatch1[%[[C1]], %[[C2]], %[[C3]]](%[[SPLAT1]][%[[C0]] to %[[C100]] for %[[C100]]], %[[SPLAT2]][%[[C0]] to %[[C100]] for %[[C100]]])
    %6 = stream.async.dispatch @executable::@dispatch1[%c1, %c2, %c3](%0[%c0 to %c100 for %c100], %1[%c0 to %c100 for %c100]) : (!stream.resource<*>{%c100}, !stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
    scf.yield %6 : !stream.resource<*>
  }
  util.return %4, %3 : !stream.resource<*>, !stream.resource<*>
}

// -----

// CHECK-LABEL: @SplatAlreadyAtSinkLocation
util.func private @SplatAlreadyAtSinkLocation(
  %arg0: i1, %arg1: i1,
  %arg2: !stream.resource<*>,
  %arg3: !stream.resource<*>,
  %arg4: !stream.resource<*>
) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c121_i32 = arith.constant 121 : i32
  // The splat is already where we would sink it to -- this used to trigger
  // infinite pattern recursion.
  // CHECK: %[[SPLAT100:.+]] = stream.async.splat %c121_i32 : i32 -> !stream.resource<*>{%c100}
  // CHECK-NEXT: %[[SPLAT101:.+]] = stream.async.splat %c121_i32 : i32 -> !stream.resource<*>{%c101}
  // CHECK-NEXT: cf.cond_br %arg1, ^bb1, ^bb2
  %0 = stream.async.splat %c121_i32 : i32 -> !stream.resource<*>{%c100}
  %1 = stream.async.splat %c121_i32 : i32 -> !stream.resource<*>{%c101}
  cf.cond_br %arg1, ^bb1, ^bb2
// CHECK: ^bb1:
^bb1:
  // CHECK: stream.async.dispatch @executable::@dispatch0[%c1, %c2, %c3](%[[SPLAT100]][%c0 to %c100 for %c100], %[[SPLAT101]][%c0 to %c101 for %c101]) : (!stream.resource<*>{%c100}, !stream.resource<*>{%c101}) -> !stream.resource<*>{%c100}
  %2 = stream.async.dispatch @executable::@dispatch0[%c1, %c2, %c3](%0[%c0 to %c100 for %c100], %1[%c0 to %c101 for %c101]) : (!stream.resource<*>{%c100}, !stream.resource<*>{%c101}) -> !stream.resource<*>{%c100}
  cf.br ^bb3(%2 : !stream.resource<*>)
// CHECK: ^bb2:
^bb2:
  // CHECK: stream.async.dispatch @executable::@dispatch1[%c1, %c2, %c3](%[[SPLAT100]][%c0 to %c100 for %c100], %[[SPLAT101]][%c0 to %c101 for %c101]) : (!stream.resource<*>{%c100}, !stream.resource<*>{%c101}) -> !stream.resource<*>{%c100}
  %3 = stream.async.dispatch @executable::@dispatch1[%c1, %c2, %c3](%0[%c0 to %c100 for %c100], %1[%c0 to %c101 for %c101]) : (!stream.resource<*>{%c100}, !stream.resource<*>{%c101}) -> !stream.resource<*>{%c100}
  cf.br ^bb3(%3 : !stream.resource<*>)
// CHECK: ^bb3(
^bb3(%arg6: !stream.resource<*>):
  util.return %arg6 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @PropagateClonableOps
util.func private @PropagateClonableOps(%arg0: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[T:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%arg0}
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%arg0}
  // CHECK-NOT: stream.async.clone
  %1 = stream.async.clone %0 : !stream.resource<*>{%arg0} -> !stream.resource<*>{%arg0}
  // CHECK: util.return %[[T]]
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @ConvertSplatConstantsIntoSplats
util.func private @ConvertSplatConstantsIntoSplats(%arg0: index) -> (!stream.resource<transient>, !stream.resource<transient>) {
  // CHECK: %[[CST:.+]] = arith.constant 3 : i32
  // CHECK: = stream.async.constant : !stream.resource<transient>{%arg0} = dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
  %0 = stream.async.constant : !stream.resource<transient>{%arg0} = dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
  // CHECK-NOT: = stream.async.constant : !stream.resource<transient>{%arg0} = dense<[3]> : tensor<8xi32>
  // CHECK: = stream.async.splat %[[CST]] : i32 -> !stream.resource<transient>{%arg0}
  %1 = stream.async.constant : !stream.resource<transient>{%arg0} = dense<3> : tensor<8xi32>
  util.return %0, %1 : !stream.resource<transient>, !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncSliceOp
util.func private @FoldAsyncSliceOp(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.async.slice
  %0 = stream.async.slice %arg0[%c0 to %arg1] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
  // CHECK: util.return %arg0
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @PropagateSplatsThroughSlices
util.func private @PropagateSplatsThroughSlices(%arg0: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[T:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c128}
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%arg0}
  // CHECK-NOT: stream.async.slice
  %1 = stream.async.slice %0[%c0 to %c128] : !stream.resource<*>{%arg0} -> !stream.resource<*>{%c128}
  // CHECK: util.return %[[T]]
  util.return %1 : !stream.resource<*>
}

// -----

// Allow pattern because we can verify the target is safe to elide.

// CHECK-LABEL: @FlattenFullFillToSplat
util.func private @FlattenFullFillToSplat(%arg0: index, %arg1: i32) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%arg0}
  // CHECK: %[[T:.+]] = stream.async.splat %arg1 : i32 -> !stream.resource<*>{%arg0}
  %0 = stream.async.fill %arg1, %target[%c0 to %arg0 for %arg0] : i32 -> %target as !stream.resource<*>{%arg0}
  // CHECK: util.return %[[T]]
  util.return %0 : !stream.resource<*>
}

// -----

// The target is tied and we cannot avoid the fill.

// CHECK-LABEL: @FlattenFullFillToSplatUnsafe
util.func private @FlattenFullFillToSplatUnsafe(%arg0: index, %arg1: i32, %arg2: !hal.buffer_view) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: stream.tensor.import
  %target = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<8xi32> in !stream.resource<*>{%arg0}
  // CHECK: stream.async.fill
  %0 = stream.async.fill %arg1, %target[%c0 to %arg0 for %arg0] : i32 -> %target as !stream.resource<*>{%arg0}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @ElideRedundantFill
util.func private @ElideRedundantFill(%arg0: !stream.resource<*>, %arg1: index, %arg2: i32) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[T:.+]] = stream.async.splat %arg2 : i32 -> !stream.resource<*>{%arg1}
  %0 = stream.async.splat %arg2 : i32 -> !stream.resource<*>{%arg1}
  // CHECK-NOT: stream.async.fill
  %1 = stream.async.fill %arg2, %0[%c0 to %arg1 for %arg1] : i32 -> %0 as !stream.resource<*>{%arg1}
  // CHECK: util.return %[[T]]
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @ElideRedundantFillBitPatterns
util.func private @ElideRedundantFillBitPatterns(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[CD_I8:.+]] = arith.constant -51 : i8
  %cCDCD_i16 = arith.constant 0xCDCD : i16
  %cCDCDCDCD_i32 = arith.constant 0xCDCDCDCD : i32
  // CHECK: %[[T:.+]] = stream.async.splat %[[CD_I8]] : i8 -> !stream.resource<*>{%arg1}
  %0 = stream.async.splat %cCDCDCDCD_i32 : i32 -> !stream.resource<*>{%arg1}
  // CHECK-NOT: stream.async.fill
  %1 = stream.async.fill %cCDCD_i16, %0[%c0 to %arg1 for %arg1] : i16 -> %0 as !stream.resource<*>{%arg1}
  // CHECK: util.return %[[T]]
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @CoalesceAdjacentFills
util.func private @CoalesceAdjacentFills(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c20 = arith.constant 20 : index
  %c0_i8 = arith.constant 0 : i8
  %c1_i8 = arith.constant 1 : i8
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[FILL_0:.+]] = stream.async.fill %c0_i8, %arg0[%c4 to %c16 for %c12] : i8 -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.fill %c0_i8, %arg0[%c4 to %c8 for %c4] : i8 -> %arg0 as !stream.resource<*>{%arg1}
  %1 = stream.async.fill %c0_i32, %0[%c8 to %c12 for %c4] : i32 -> %0 as !stream.resource<*>{%arg1}
  %2 = stream.async.fill %c0_i8, %1[%c12 to %c16 for %c4] : i8 -> %1 as !stream.resource<*>{%arg1}
  // CHECK: %[[FILL_1:.+]] = stream.async.fill %c1_i8, %[[FILL_0]][%c16 to %c20 for %c4] : i8 -> %[[FILL_0]] as !stream.resource<*>{%arg1}
  %3 = stream.async.fill %c1_i8, %2[%c16 to %c20 for %c4] : i8 -> %2 as !stream.resource<*>{%arg1}
  // CHECK: util.return %[[FILL_1]]
  util.return %3 : !stream.resource<*>
}

// -----

// If we can't analyze the resources we can't fold as the update may be required
// to preserve an in-place update of an external resource.

// CHECK-LABEL: @DontFoldNonLocalAsyncUpdateOp
util.func private @DontFoldNonLocalAsyncUpdateOp(%arg0: !stream.resource<*>, %arg1: !stream.resource<*>, %arg2: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: stream.async.update
  %0 = stream.async.update %arg1, %arg0[%c0 to %arg2] : !stream.resource<*>{%arg2} -> %arg0 as !stream.resource<*>{%arg2}
  util.return %0 : !stream.resource<*>
}

// -----

// We can only fold when we prove that the target has value semantics.

// CHECK-LABEL: @FoldLocalAsyncUpdateOp
util.func private @FoldLocalAsyncUpdateOp(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%arg1}
  // CHECK-NOT: stream.async.update
  %update = stream.async.update %arg0, %splat[%c0 to %arg1] : !stream.resource<*>{%arg1} -> %result as !stream.resource<*>{%arg1}
  // CHECK: util.return %arg0
  util.return %update : !stream.resource<*>
}

// -----

// Tests that updates of a value into itself are no-oped.

// CHECK-LABEL: @FoldNoOpAsyncUpdateOp
util.func private @FoldNoOpAsyncUpdateOp(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.async.update
  %0 = stream.async.update %arg0, %arg0[%c0 to %arg1] : !stream.resource<*>{%arg1} -> %arg0 as !stream.resource<*>{%arg1}
  // CHECK: util.return %arg0
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @ElideInPlaceUpdateUpdate
util.func private @ElideInPlaceUpdateUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = stream.async.update %arg0, %arg2[%c0 to %arg1] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg3}
  %0 = stream.async.update %arg0, %arg2[%c0 to %arg1] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg3}
  // CHECK-NOT: stream.async.update
  %1 = stream.async.update %0, %arg2[%c0 to %arg3] : !stream.resource<*>{%arg3} -> %arg2 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[RESULT]]
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @ElideInPlaceUpdateDispatch
util.func private @ElideInPlaceUpdateDispatch(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex::@fn(%arg0[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %arg0{%arg1}
  %0 = stream.async.dispatch @ex::@fn(%arg0[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %arg0{%arg1}
  // CHECK-NOT: stream.async.update
  %1 = stream.async.update %0, %arg0[%c0 to %arg1] : !stream.resource<*>{%arg1} -> %arg0 as !stream.resource<*>{%arg1}
  // CHECK: util.return %[[RESULT]]
  util.return %1 : !stream.resource<*>
}

// -----

// Tests that multiple users of the produced value will still allow the update
// to be elided so long as they are reads.

// CHECK-LABEL: @ElideInPlaceUpdateDispatchMultiUse
util.func private @ElideInPlaceUpdateDispatchMultiUse(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESULT0:.+]] = stream.async.dispatch @ex::@fn0(%arg0[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %arg0{%arg1}
  %0 = stream.async.dispatch @ex::@fn0(%arg0[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %arg0{%arg1}
  // CHECK-NOT: stream.async.update
  %1 = stream.async.update %0, %arg0[%c0 to %arg1] : !stream.resource<*>{%arg1} -> %arg0 as !stream.resource<*>{%arg1}
  // CHECK: %[[RESULT1:.+]] = stream.async.dispatch @ex::@fn1(%[[RESULT0]][%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
  %2 = stream.async.dispatch @ex::@fn1(%0[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
  // CHECK: util.return %[[RESULT0]], %[[RESULT1]]
  util.return %1, %2 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that writes on the update source will fail to elide the update.
// TODO(benvanik): support looking for writes only prior to the update that are
// known-safe.

// CHECK-LABEL: @ElideInPlaceUpdateDispatchMultiUseWrite
util.func private @ElideInPlaceUpdateDispatchMultiUseWrite(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  // CHECK: stream.async.dispatch @ex::@fn0
  %0 = stream.async.dispatch @ex::@fn0(%arg0[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %arg0{%arg1}
  // CHECK: stream.async.update
  %1 = stream.async.update %0, %arg0[%c0 to %arg1] : !stream.resource<*>{%arg1} -> %arg0 as !stream.resource<*>{%arg1}
  // CHECK: stream.async.dispatch @ex::@fn1
  %2 = stream.async.dispatch @ex::@fn1(%0[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %0{%arg1}
  util.return %1, %2 : !stream.resource<*>, !stream.resource<*>
}

// -----

// CHECK-LABEL: @CombineSplatUpdateFromToFill
util.func private @CombineSplatUpdateFromToFill(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.splat
  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c128}
  // CHECK: %[[T:.+]] = stream.async.fill %c123_i32, %arg0[%c0 to %c128 for %c128] : i32 -> %arg0 as !stream.resource<*>{%arg1}
  %1 = stream.async.update %0, %arg0[%c0 to %c128] : !stream.resource<*>{%c128} -> %arg0 as !stream.resource<*>{%arg1}
  // CHECK: util.return %[[T]]
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @CombineSliceUpdateFromToCopy
util.func private @CombineSliceUpdateFromToCopy(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: stream.async.slice
  %0 = stream.async.slice %arg0[%c0 to %c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
  // CHECK: %[[T:.+]] = stream.async.copy %arg0[%c0 to %c128], %arg2[%c0 to %c128], %c128 : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg3}
  %1 = stream.async.update %0, %arg2[%c0 to %c128] : !stream.resource<*>{%c128} -> %arg2 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[T]]
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @AsyncCopyFullSourceToUpdate
util.func private @AsyncCopyFullSourceToUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // This copy is from the full source (0..%arg3) so it can be turned into an update.
  // CHECK: = stream.async.update %arg2, %arg0[%c0 to %arg3] : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.copy %arg2[%c0 to %arg3], %arg0[%c0 to %arg3], %arg3 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}

  // This copy is only a partial section of the source and needs to remain a copy.
  // CHECK: = stream.async.copy %arg2[%c16 to %arg3], %arg0[%c0 to %arg3], %c8 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  %1 = stream.async.copy %arg2[%c16 to %arg3], %arg0[%c0 to %arg3], %c8 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}

  util.return %0, %1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// CHECK-LABEL: @FoldAsyncTransferOp
util.func private @FoldAsyncTransferOp(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.resource<transient> {
  // CHECK-NOT: stream.async.transfer
  %0 = stream.async.transfer %arg0 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg1}
  %1 = stream.async.transfer %0 : !stream.resource<staging>{%arg1} -> !stream.resource<transient>{%arg1}
  util.return %1 : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @RedundantTransferElision
util.func private @RedundantTransferElision(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.resource<transient> {
  // CHECK-NOT: stream.async.transfer
  %0 = stream.async.transfer %arg0 : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%arg1}
  util.return %0 : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @IntermediateTransferElision
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<constant>, %[[SIZE:.+]]: index)
util.func private @IntermediateTransferElision(%source: !stream.resource<constant>, %size: index) -> !stream.resource<external> {
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[SOURCE]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %transfer0 = stream.async.transfer %source : !stream.resource<constant>{%size} -> !stream.resource<staging>{%size}
  // CHECK-NOT: stream.async.transfer
  %transfer1 = stream.async.transfer %transfer0 : !stream.resource<staging>{%size} -> !stream.resource<external>{%size}
  // CHECK-NEXT: util.return %[[TRANSFER]]
  util.return %transfer1 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @FoldAsyncLoadBitcast
util.func private @FoldAsyncLoadBitcast(%arg0: !stream.resource<staging>, %arg1: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[F32:.+]] = stream.async.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> f32
  %0 = stream.async.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> i32
  // CHECK-NOT: arith.bitcast
  %1 = arith.bitcast %0 : i32 to f32
  // CHECK: util.return %[[F32]]
  util.return %1 : f32
}

// -----

// CHECK-LABEL: @FoldAsyncStoreBitcast
util.func private @FoldAsyncStoreBitcast(%arg0: !stream.resource<staging>, %arg1: index, %arg2: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  %0 = arith.bitcast %arg2 : f32 to i32
  // CHECK: = stream.async.store %arg2, %arg0[%c0] : f32 -> %arg0 as !stream.resource<staging>{%arg1}
  %1 = stream.async.store %0, %arg0[%c0] : i32 -> %arg0 as !stream.resource<staging>{%arg1}
  util.return %1 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @ElideImmediateAsyncExecuteWaits
util.func private @ElideImmediateAsyncExecuteWaits(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: stream.async.execute with
  %0:2 = stream.async.execute await(%imm) => with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> %arg0 as !stream.resource<*>{%arg1} {
    // CHECK: stream.async.dispatch
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg2[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK: stream.yield
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ChainAsyncExecuteWaits
util.func private @ChainAsyncExecuteWaits(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: stream.timepoint.await
  %0 = stream.timepoint.await %arg2 => %arg0 : !stream.resource<*>{%arg1}
  // CHECK: stream.async.execute await(%arg2) => with
  %1:2 = stream.async.execute with(%0 as %arg3: !stream.resource<*>{%arg1}) -> %0 as !stream.resource<*>{%arg1} {
    // CHECK: stream.async.dispatch
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK: stream.yield
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  util.return %1#0, %1#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @CloneCapturedAsyncExecuteSubviewOps
util.func private @CloneCapturedAsyncExecuteSubviewOps(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: stream.resource.subview
  %0 = stream.resource.subview %arg0[%c0] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
  // CHECK: = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> %arg0{%c128}
  %1:2 = stream.async.execute with(%0 as %arg3: !stream.resource<*>{%c128}) -> %0{%c128} {
    // CHECK: %[[T:.+]] = stream.resource.subview %arg2[%c0] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
    // CHECK: stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%[[T]]
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK: stream.yield
    stream.yield %1 : !stream.resource<*>{%c128}
  } => !stream.timepoint
  util.return %1#0, %1#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideNoOpAsyncExecuteOp
util.func private @ElideNoOpAsyncExecuteOp(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK-NOT: stream.async.execute
  %1:2 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> %arg0{%arg1} {
    stream.yield %arg3 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  // CHECK: util.return %arg0, %[[IMM]]
  util.return %1#0, %1#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @TieRegionResultsAsyncExecuteOp
util.func private @TieRegionResultsAsyncExecuteOp(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> %arg0{%arg1}
  %0:2 = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
    // CHECK: %[[T:.+]] = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg2
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg2[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %arg2{%arg1}
    // CHECK: stream.yield %[[T]]
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideUnusedAsyncExecuteOp
util.func private @ElideUnusedAsyncExecuteOp(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: stream.async.execute
  %0:2 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  util.return
}


// -----

// CHECK-LABEL: @FoldAsyncExecuteDuplicateResults
// CHECK-SAME: (%[[SPLAT_A_SIZE:.+]]: index, %[[SPLAT_A_VALUE:.+]]: i32, %[[SPLAT_B_SIZE:.+]]: index, %[[SPLAT_B_VALUE:.+]]: i32)
util.func private @FoldAsyncExecuteDuplicateResults(%splat_a_size: index, %splat_a_value: i32, %splat_b_size: index, %splat_b_value: i32) -> (!stream.resource<*>, !stream.resource<*>, !stream.resource<*>, !stream.timepoint) {
  // CHECK: %[[RESULTS:.+]]:2, %[[TIMEPOINT:.+]] = stream.async.execute with() -> (!stream.resource<*>{%[[SPLAT_A_SIZE]]}, !stream.resource<*>{%[[SPLAT_B_SIZE]]}) {
  %results:3, %timepoint = stream.async.execute with() -> (!stream.resource<*>{%splat_a_size}, !stream.resource<*>{%splat_b_size}, !stream.resource<*>{%splat_a_size}) {
    // CHECK: %[[SPLAT_A:.+]] = stream.async.splat %[[SPLAT_A_VALUE]]
    %splat_a = stream.async.splat %splat_a_value : i32 -> !stream.resource<*>{%splat_a_size}
    // CHECK: %[[SPLAT_B:.+]] = stream.async.splat %[[SPLAT_B_VALUE]]
    %splat_b = stream.async.splat %splat_b_value : i32 -> !stream.resource<*>{%splat_b_size}
    // CHECK: stream.yield %[[SPLAT_A]], %[[SPLAT_B]] : !stream.resource<*>{%[[SPLAT_A_SIZE]]}, !stream.resource<*>{%[[SPLAT_B_SIZE]]}
    stream.yield %splat_a, %splat_b, %splat_a : !stream.resource<*>{%splat_a_size}, !stream.resource<*>{%splat_b_size}, !stream.resource<*>{%splat_a_size}
  } => !stream.timepoint
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULTS]]#1, %[[RESULTS]]#0, %[[TIMEPOINT]]
  util.return %results#0, %results#1, %results#2, %timepoint : !stream.resource<*>, !stream.resource<*>, !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldAsyncExecuteTiedDuplicateResults
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<*>, %[[TARGET_SIZE:.+]]: index, %[[SPLAT_SIZE:.+]]: index, %[[SPLAT_VALUE:.+]]: i32)
util.func private @FoldAsyncExecuteTiedDuplicateResults(%target: !stream.resource<*>, %target_size: index, %splat_size: index, %splat_value: i32) -> (!stream.resource<*>, !stream.resource<*>, !stream.resource<*>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[RESULTS:.+]]:2, %[[TIMEPOINT:.+]] = stream.async.execute with({{.+}}) -> (%[[TARGET]]{%[[TARGET_SIZE]]}, !stream.resource<*>{%[[SPLAT_SIZE]]}) {
  %results:3, %timepoint = stream.async.execute with(%target as %target_capture: !stream.resource<*>{%target_size}) -> (%target as !stream.resource<*>{%target_size}, !stream.resource<*>{%splat_size}, %target as !stream.resource<*>{%target_size}) {
    // CHECK: %[[TARGET_FILL:.+]] = stream.async.fill
    %target_fill = stream.async.fill %splat_value, %target_capture[%c0 to %c128 for %c128] : i32 -> %target_capture as !stream.resource<*>{%target_size}
    // CHECK: %[[SPLAT:.+]] = stream.async.splat %[[SPLAT_VALUE]]
    %splat = stream.async.splat %splat_value : i32 -> !stream.resource<*>{%splat_size}
    // CHECK: stream.yield %[[TARGET_FILL]], %[[SPLAT]] : !stream.resource<*>{%[[TARGET_SIZE]]}, !stream.resource<*>{%[[SPLAT_SIZE]]}
    stream.yield %target_fill, %splat, %target_fill : !stream.resource<*>{%target_size}, !stream.resource<*>{%splat_size}, !stream.resource<*>{%target_size}
  } => !stream.timepoint
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULTS]]#1, %[[RESULTS]]#0, %[[TIMEPOINT]]
  util.return %results#0, %results#1, %results#2, %timepoint : !stream.resource<*>, !stream.resource<*>, !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @TieRegionResultsAsyncConcurrentOp
util.func private @TieRegionResultsAsyncConcurrentOp(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> %arg0{%arg1}
  %0:2 = stream.async.execute with(%arg0 as %arg2: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
    // CHECK: %[[EXEC_T:.+]] = stream.async.concurrent with(%arg2 as %arg3: !stream.resource<*>{%arg1}) -> %arg2{%arg1}
    %1 = stream.async.concurrent with(%arg2 as %arg3: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
      // CHECK: %[[WAVE_T:.+]] = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %arg3{%arg1}
      %2 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> %arg3{%arg1}
      // CHECK: stream.yield %[[WAVE_T]]
      stream.yield %2 : !stream.resource<*>{%arg1}
    }
    // CHECK: stream.yield %[[EXEC_T]]
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideUnusedAsyncConcurrentOp
util.func private @ElideUnusedAsyncConcurrentOp(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: stream.async.execute
  %0:2 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
    // CHECK: stream.async.dispatch @executable::@dispatch0
    %1 = stream.async.dispatch @executable::@dispatch0[%c1, %c1, %c1](%arg3[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK-NOT: stream.async.concurrent
    %2 = stream.async.concurrent with(%arg3 as %arg4: !stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1} {
      // CHECK-NOT: stream.async.dispatch @executable::@dispatch1
      %3 = stream.async.dispatch @executable::@dispatch1[%c1, %c1, %c1](%arg4[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
      stream.yield %3 : !stream.resource<*>{%arg1}
    }
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}
