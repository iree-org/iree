// RUN: iree-opt --split-input-file --canonicalize --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @FoldTimepointExport
util.func private @FoldTimepointExport(%arg0: !hal.semaphore, %arg1: index) -> (!hal.semaphore, index) {
  // CHECK-NOT: stream.timepoint.import
  %0 = stream.timepoint.import %arg0, %arg1 : (!hal.semaphore, index) => !stream.timepoint
  // CHECK-NOT: stream.timepoint.export
  %1:2 = stream.timepoint.export %0 => (!hal.semaphore, index)
  // CHECK: util.return %arg0, %arg1
  util.return %1#0, %1#1 : !hal.semaphore, index
}

// -----

// CHECK-LABEL: @DontFoldTimepointExportMismatch
util.func private @DontFoldTimepointExportMismatch(%arg0: !hal.semaphore, %arg1: index) -> (!hal.semaphore, i32) {
  // CHECK: stream.timepoint.import
  %0 = stream.timepoint.import %arg0, %arg1 : (!hal.semaphore, index) => !stream.timepoint
  // CHECK-NEXT: stream.timepoint.export
  %1:2 = stream.timepoint.export %0 => (!hal.semaphore, i32)
  util.return %1#0, %1#1 : !hal.semaphore, i32
}

// -----

// CHECK-LABEL: @PassThroughChainExternal
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[ARG_FENCE:.+]]: !hal.fence)
util.func private @PassThroughChainExternal(%device: !hal.device, %arg_fence: !hal.fence) -> !hal.fence {
  // CHECK-NOT: stream.timepoint.import
  %timepoint = stream.timepoint.import %arg_fence : (!hal.fence) => !stream.timepoint
  // CHECK-NOT: hal.fence.create
  %chained_fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
  // CHECK-NOT: stream.timepoint.chain_external
  stream.timepoint.chain_external %timepoint => (%chained_fence : !hal.fence)
  // CHECK: util.return %[[ARG_FENCE]]
  util.return %chained_fence : !hal.fence
}

// -----

// Tests that external chained values we can't analyze aren't replaced.

// CHECK-LABEL: @DontPassThroughChainExternal
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[ARG_FENCE:.+]]: !hal.fence, %[[CHAINED_FENCE:.+]]: !hal.fence)
util.func private @DontPassThroughChainExternal(%device: !hal.device, %arg_fence: !hal.fence, %chained_fence: !hal.fence) -> !hal.fence {
  // CHECK: %[[TIMEPOINT:.+]] = stream.timepoint.import %[[ARG_FENCE]]
  %timepoint = stream.timepoint.import %arg_fence : (!hal.fence) => !stream.timepoint
  // CHECK: stream.timepoint.chain_external %[[TIMEPOINT]] => (%[[CHAINED_FENCE]]
  stream.timepoint.chain_external %timepoint => (%chained_fence : !hal.fence)
  // CHECK: util.return %[[CHAINED_FENCE]]
  util.return %chained_fence : !hal.fence
}

// -----

// CHECK-LABEL: @FoldTimepointJoinOp
util.func private @FoldTimepointJoinOp(%arg0: !stream.timepoint) -> !stream.timepoint {
  // CHECK-NOT: stream.timepoint.join
  %0 = stream.timepoint.join max(%arg0) => !stream.timepoint
  // CHECK: util.return %arg0
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateTimepointJoinOperands
util.func private @ElideImmediateTimepointJoinOperands(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> !stream.timepoint {
  %0 = stream.timepoint.immediate => !stream.timepoint
  %1 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: = stream.timepoint.join max(%arg0, %arg1)
  %2 = stream.timepoint.join max(%arg0, %0, %1, %arg1) => !stream.timepoint
  util.return %2 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateTimepointJoinOperandsAll
util.func private @ElideImmediateTimepointJoinOperandsAll() -> !stream.timepoint {
  %0 = stream.timepoint.immediate => !stream.timepoint
  %1 = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: stream.timepoint.join
  %2 = stream.timepoint.join max(%0, %1) => !stream.timepoint
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  // CHECK: util.return %[[IMM]]
  util.return %2 : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldDuplicateTimepointJoinOperands
util.func private @FoldDuplicateTimepointJoinOperands(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: = stream.timepoint.join max(%arg0, %arg1)
  %0 = stream.timepoint.join max(%arg0, %arg1, %arg0, %arg1) => !stream.timepoint
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ExpandTimepointJoinOperands
util.func private @ExpandTimepointJoinOperands(%arg0: !stream.timepoint, %arg1: !stream.timepoint, %arg2: !stream.timepoint, %arg3: !stream.timepoint) -> !stream.timepoint {
  %join0 = stream.timepoint.join max(%arg0, %arg1) => !stream.timepoint
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%arg2, %arg0, %arg1, %arg3)
  %join1 = stream.timepoint.join max(%arg2, %join0, %arg3) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join1 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateBarrier
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func private @ElideImmediateBarrier(%size: index) -> (!stream.resource<external>, !stream.timepoint) {
  // CHECK-DAG: %[[RESOURCE:.+]] = stream.resource.alloc
  %r0 = stream.resource.alloc uninitialized : !stream.resource<external>{%size}
  // CHECK-DAG: %[[FENCE:.+]] = stream.timepoint.immediate
  // CHECK-NOT: stream.timepoint.barrier
  %r1, %r1t = stream.timepoint.barrier %r0 : !stream.resource<external>{%size} => !stream.timepoint
  // CHECK: util.return %[[RESOURCE]], %[[FENCE]]
  util.return %r1, %r1t : !stream.resource<external>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ChainTimepoints
// CHECK-SAME: (%[[FENCE:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<external>)
util.func private @ChainTimepoints(%fence: !stream.timepoint, %source: !stream.resource<external>) -> (!stream.resource<external>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: stream.timepoint.await
  %r0 = stream.timepoint.await %fence => %source : !stream.resource<external>{%c128}
  // CHECK-NOT: stream.timepoint.barrier
  %r1, %r1t = stream.timepoint.barrier %r0 : !stream.resource<external>{%c128} => !stream.timepoint
  // CHECK: util.return %[[SOURCE]], %[[FENCE]]
  util.return %r1, %r1t : !stream.resource<external>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateHostAwaits
util.func private @ElideImmediateHostAwaits(%arg0: !stream.resource<staging>) -> !stream.resource<staging> {
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.timepoint.immediate
  %0 = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: stream.timepoint.await
  %1 = stream.timepoint.await %0 => %arg0 : !stream.resource<staging>{%c100}
  // CHECK: util.return %arg0
  util.return %1 : !stream.resource<staging>
}

// -----

// Ensures that the await moves to the first common dominator of bb2/bb3 that
// use the awaited resources.

// CHECK-LABEL: @SinkAwaitToFirstConsumer
util.func private @SinkAwaitToFirstConsumer(
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
  // CHECK: cf.cond_br %arg0, ^bb1, ^bb4
  cf.cond_br %arg0, ^bb1, ^bb4(%arg4 : !stream.resource<external>)
// CHECK: ^bb1:
^bb1:
  // CHECK: %[[READY:.+]]:2 = stream.timepoint.await %arg5 => %arg2, %arg3 : !stream.resource<constant>{%c100}, !stream.resource<staging>{%c200}
  // CHECK-NEXT: cf.cond_br %arg1, ^bb2, ^bb3
  cf.cond_br %arg1, ^bb2, ^bb3
// CHECK: ^bb2:
^bb2:
  // CHECK: = stream.async.clone %[[READY]]#0
  %1 = stream.async.transfer %0#0 : !stream.resource<constant>{%c100} -> !stream.resource<external>{%c100}
  cf.br ^bb4(%1 : !stream.resource<external>)
// CHECK: ^bb3:
^bb3:
  // CHECK: = stream.async.transfer %[[READY]]#1
  %2 = stream.async.transfer %0#1 : !stream.resource<staging>{%c200} -> !stream.resource<external>{%c200}
  cf.br ^bb4(%2 : !stream.resource<external>)
// CHECK: ^bb4(
^bb4(%arg6: !stream.resource<external>):
  util.return %arg6 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @SinkAwaitToFirstConsumerRegion
util.func private @SinkAwaitToFirstConsumerRegion(
  %arg0: i1, %arg1: i1, %arg2: i1,
  %arg3: !stream.resource<constant>,
  %arg4: !stream.resource<staging>,
  %arg5: !stream.resource<external>,
  %arg6: !stream.timepoint
) -> (!stream.resource<external>, !stream.resource<external>) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-NOT stream.timepoint.await
  %0:2 = stream.timepoint.await %arg6 => %arg3, %arg4 : !stream.resource<constant>{%c100}, !stream.resource<staging>{%c200}
  cf.cond_br %arg0, ^bb1, ^bb2(%arg5, %arg5 : !stream.resource<external>,  !stream.resource<external>)
  // CHECK: ^bb1
  ^bb1:
  // CHECK: stream.timepoint.await
  // CHECK: "fake.region"
  %3 = "fake.region"() ({
    // CHECK: "fake.region"
    %4 = "fake.region"() ({
      // CHECK: stream.async.clone
      %5 = stream.async.transfer %0#0 : !stream.resource<constant>{%c100} -> !stream.resource<external>{%c100}
      // CHECK: "fake.yield"
      "fake.yield"(%5) : (!stream.resource<external>) -> ()
    }) : () -> (!stream.resource<external>)
    // CHECK: "fake.yield"
    "fake.yield"(%4) : (!stream.resource<external>) -> ()
  }) : () -> (!stream.resource<external>)
  // CHECK: stream.async.transfer
  %2 = stream.async.transfer %0#1 : !stream.resource<staging>{%c200} -> !stream.resource<external>{%c200}
  // CHECK: cf.br
  cf.br ^bb2(%2, %3 : !stream.resource<external>, !stream.resource<external>)
  // CHECK: ^bb2
^bb2(%arg7: !stream.resource<external>, %arg8: !stream.resource<external>):
  // CHECK:util.return
  util.return %arg7, %arg8 : !stream.resource<external>,  !stream.resource<external>
}

// -----

// CHECK-LABEL: @SinkSubviewsAcrossAwaits
util.func private @SinkSubviewsAcrossAwaits(
  %arg0: !stream.resource<*>, %arg1: index,
  %arg2: !stream.timepoint
) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK: %[[READY:.+]] = stream.timepoint.await %arg2 => %arg0 : !stream.resource<*>{%arg1}
  // CHECK: %[[RET:.+]] = stream.resource.subview %[[READY]][%c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c256}
  %0 = stream.resource.subview %arg0[%c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c256}
  %1 = stream.timepoint.await %arg2 => %0 : !stream.resource<*>{%c256}
  // CHECK: util.return %[[RET]]
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @GroupAwaitsByTimepoint
util.func private @GroupAwaitsByTimepoint(
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
  // CHECK-NEXT: util.return %[[RET]]#0, %[[RET]]#1, %[[RET]]#2, %[[RET]]#3
  util.return %0, %1, %2#0, %2#1 : !stream.resource<*>, !stream.resource<*>, !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that the pattern doesn't kick in when it would be unsafe to group the
// awaits due to operand dependencies.

util.func private @materializeResource0() -> !stream.resource<*>
util.func private @materializeResource1(!stream.resource<*>) -> !stream.resource<*>

// CHECK-LABEL: @GroupAwaitsByTimepointUnsafe
util.func private @GroupAwaitsByTimepointUnsafe(
  %arg0: !stream.timepoint
) -> (!stream.resource<*>, !stream.resource<*>) {
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  // CHECK: util.call @materializeResource0
  %r0a = util.call @materializeResource0() : () -> !stream.resource<*>
  // CHECK-NEXT: stream.timepoint.await
  %r0b = stream.timepoint.await %arg0 => %r0a : !stream.resource<*>{%c100}
  // CHECK-NEXT: util.call @materializeResource1
  %r1a = util.call @materializeResource1(%r0b) : (!stream.resource<*>) -> !stream.resource<*>
  // CHECK-NEXT: stream.timepoint.await
  %r1b = stream.timepoint.await %arg0 => %r1a : !stream.resource<*>{%c101}
  // CHECK-NEXT: util.return
  util.return %r0b, %r1b : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that the pattern doesn't kick in when the same timepoint are waited in
// different blocks.

util.func private @materializeResource() -> !stream.resource<*>

// CHECK-LABEL: @DontGroupAwaitsByTimepointAcrossBlocks
util.func private @DontGroupAwaitsByTimepointAcrossBlocks(
  %arg0: !stream.timepoint,
  %arg1: !stream.resource<*>,
  %arg2: i1
) -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %0 = stream.timepoint.await %arg0 => %arg1 : !stream.resource<*>{%c100}
  // CHECK: cf.cond_br
  cf.cond_br %arg2, ^bb0, ^bb1
// CHECK: ^bb
^bb0:
  // CHECK: stream.timepoint.await %arg0 => %arg1
  util.return %0 : !stream.resource<*>
// CHECK: ^bb
^bb1:
  // CHECK: %[[R:.+]] = util.call @materializeResource
  %r = util.call @materializeResource() : () -> !stream.resource<*>
  // CHECK: stream.timepoint.await %arg0 => %[[R]]
  %1 = stream.timepoint.await %arg0 => %r : !stream.resource<*>{%c101}
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @FoldDuplicateAwaitResources
util.func private @FoldDuplicateAwaitResources(
  %arg0: !stream.timepoint,
  %arg1: !stream.resource<staging>, %arg2: !stream.resource<*>
) -> (!stream.resource<staging>, !stream.resource<*>, !stream.resource<staging>, !stream.resource<staging>) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[RET:.+]]:2 = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  %0:4 = stream.timepoint.await %arg0 => %arg1, %arg2, %arg1, %arg1 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}, !stream.resource<staging>{%c100}, !stream.resource<staging>{%c100}
  // CHECK: util.return %[[RET]]#0, %[[RET]]#1, %[[RET]]#0, %[[RET]]#0
  util.return %0#0, %0#1, %0#2, %0#3 : !stream.resource<staging>, !stream.resource<*>, !stream.resource<staging>, !stream.resource<staging>
}

// -----

// CHECK-LABEL: @ElideUnusedTimepointAwaitOp
util.func private @ElideUnusedTimepointAwaitOp(
  %arg0: !stream.timepoint,
  %arg1: !stream.resource<staging>, %arg2: !stream.resource<*>
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-NOT: stream.timepoint.await
  %0:2 = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  util.return
}
