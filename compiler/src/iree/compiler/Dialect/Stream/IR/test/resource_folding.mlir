// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldResourceSizeOp
util.func private @FoldResourceSizeOp(%arg0: !stream.resource<staging>, %arg1: index) -> (index, i32) {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.resource.size
  %0 = stream.resource.size %arg0 : !stream.resource<staging>
  // CHECK: %[[LOAD:.+]] = stream.resource.load
  %1 = stream.resource.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> i32
  // CHECK: util.return %arg1, %[[LOAD]]
  util.return %0, %1 : index, i32
}

// -----

// CHECK-LABEL: @SelectResourceSizeOp
util.func private @SelectResourceSizeOp(%arg0: !stream.resource<staging>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: i1) -> (!stream.resource<staging>, index) {
  // CHECK: %[[ARG0_T:.+]] = stream.async.transfer %arg0 {{.+}} -> !stream.resource<*>{%[[ARG0_SZ:.+]]}
  %0 = stream.async.transfer %arg0 : !stream.resource<staging>{%arg1} -> !stream.resource<*>{%arg1}
  // CHECK: %[[ARG2_T:.+]] = stream.async.transfer %arg2 {{.+}} -> !stream.resource<*>{%[[ARG2_SZ:.+]]}
  %1 = stream.async.transfer %arg2 : !stream.resource<staging>{%arg3} -> !stream.resource<*>{%arg3}
  // CHECK: %[[RET_T:.+]] = arith.select %arg4, %[[ARG0_T]], %[[ARG2_T]] : !stream.resource<*>
  %2 = arith.select %arg4, %0, %1 : !stream.resource<*>
  // CHECK: %[[RET_SIZE:.+]] = arith.select %arg4, %[[ARG0_SZ]], %[[ARG2_SZ]] : index
  %3 = stream.resource.size %2 : !stream.resource<*>
  // CHECK: = stream.async.transfer %[[RET_T]] : !stream.resource<*>{%[[RET_SIZE]]}
  %4 = stream.async.transfer %2 : !stream.resource<*>{%3} -> !stream.resource<staging>{%3}
  util.return %4, %3 : !stream.resource<staging>, index
}

// -----

// CHECK-LABEL: @FoldSubviewIntoLoadOp
util.func private @FoldSubviewIntoLoadOp(%arg0: !stream.resource<staging>, %arg1: index) -> i32 {
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK-NOT: stream.resource.subview
  %0 = stream.resource.subview %arg0[%c128] : !stream.resource<staging>{%arg1} -> !stream.resource<staging>{%c256}
  // CHECK: = stream.resource.load %arg0[%c192] : !stream.resource<staging>{%arg1} -> i32
  %1 = stream.resource.load %0[%c64] : !stream.resource<staging>{%c256} -> i32
  util.return %1 : i32
}

// -----

// CHECK-LABEL: @DontFoldSubviewIntoLoadAcrossAwaitOp
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<staging>, %[[SIZE:.+]]: index, %[[FENCE:.+]]: !stream.timepoint)
util.func private @DontFoldSubviewIntoLoadAcrossAwaitOp(%source: !stream.resource<staging>, %size: index, %fence: !stream.timepoint) -> i32 {
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK-NOT: stream.resource.subview
  %0 = stream.resource.subview %source[%c128] : !stream.resource<staging>{%size} -> !stream.resource<staging>{%c256}
  // CHECK: %[[READY:.+]] = stream.timepoint.await %[[FENCE]] => %[[SOURCE]]
  %1 = stream.timepoint.await %fence => %0 : !stream.resource<staging>{%c256}
  // CHECK: = stream.resource.load %[[READY]][%c192] : !stream.resource<staging>{%[[SIZE]]} -> i32
  %2 = stream.resource.load %1[%c64] : !stream.resource<staging>{%c256} -> i32
  util.return %2 : i32
}

// -----

// CHECK-LABEL: @FoldSubviewIntoStoreOp
util.func private @FoldSubviewIntoStoreOp(%arg0: !stream.resource<staging>, %arg1: index) {
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.resource.subview
  %0 = stream.resource.subview %arg0[%c128] : !stream.resource<staging>{%arg1} -> !stream.resource<staging>{%c256}
  // CHECK: stream.resource.store %c123_i32, %arg0[%c192] : i32 -> !stream.resource<staging>{%arg1}
  stream.resource.store %c123_i32, %0[%c64] : i32 -> !stream.resource<staging>{%c256}
  util.return
}

// -----

// A pack with no slices folds to a zero-length slab.

// CHECK-LABEL: @FoldResourcePackOpEmpty
util.func private @FoldResourcePackOpEmpty(%allocator: !hal.allocator) -> index {
  // CHECK-NEXT: %[[ZERO_LENGTH:.+]] = arith.constant 0
  %total_length = stream.resource.pack slices({}) : index
  // CHECK-NEXT: util.return %[[ZERO_LENGTH]]
  util.return %total_length : index
}

// -----

// A pack with a single slices folds to just that slice.

// CHECK-LABEL: @FoldResourcePackOpOneSlice
// CHECK-SAME: %[[OFFSET:.+]]: index,
// CHECK-SAME: %[[SIZE:.+]]: index
util.func private @FoldResourcePackOpOneSlice(%offset: index, %size: index) -> (index, index) {
  // CHECK-NOT: stream.resource.pack
  %total_length, %offset_0 =
      stream.resource.pack
        offset(%offset)
        slices({
          [0, 4] = %size
        }) : index
  // CHECK: util.return %[[SIZE]], %[[OFFSET]]
  util.return %total_length, %offset_0 : index, index
}

// -----

// A constant zero offset operand gets dropped.

// CHECK-LABEL: @PropagateResourcePackZeroOffset
util.func private @PropagateResourcePackZeroOffset(%size : index) -> (index, index, index) {
  // CHECK-NOT: constant 0
  // CHECK-NEXT: = stream.resource.pack slices({
  %base_offset = arith.constant 0 : index
  %total_length, %offset_0, %offset_1 =
      stream.resource.pack
        offset(%base_offset)
        slices({
          [0, 4] = %size,
          [1, 2] = %size,
        }) : index
  util.return %total_length, %offset_0, %offset_1 : index, index, index
}

// -----

// A base offset operand gets propagated to returned values.

// CHECK-LABEL: @PropagateResourcePackBaseOffset
// CHECK-SAME: %[[BASE_OFFSET:.+]]: index,
// CHECK-SAME: %[[SIZE:.+]]: index
util.func private @PropagateResourcePackBaseOffset(%base_offset: index, %size : index) -> (index, index, index) {
  // CHECK-NEXT: %[[PACKED:.+]]:3 =
  %total_length, %offset_0, %offset_1 =
      // CHECK-SAME: stream.resource.pack slices({
      stream.resource.pack
        offset(%base_offset)
        slices({
          [0, 4] = %size,
          [1, 2] = %size,
        }) : index
  //      CHECK: %[[ADJUSTED_0:.+]] = arith.addi %[[BASE_OFFSET]], %[[PACKED]]#1
  // CHECK-NEXT: %[[ADJUSTED_1:.+]] = arith.addi %[[BASE_OFFSET]], %[[PACKED]]#2
  // CHECK-NEXT: util.return %[[PACKED]]#0, %[[ADJUSTED_0]], %[[ADJUSTED_1]]
  util.return %total_length, %offset_0, %offset_1 : index, index, index
}

// -----

// Intervals should be sorted.

// CHECK-LABEL: @CanonicalizeResourcePackIntervals
// CHECK-SAME: %[[SIZE:.+]]: index
util.func private @CanonicalizeResourcePackIntervals(%size : index) -> (index, index, index) {
  // CHECK-NEXT: %[[PACKED:.+]]:3 =
  %total_length, %offset_0, %offset_1 =
      // CHECK-SAME: stream.resource.pack slices({
      stream.resource.pack
        slices({
          // CHECK-NEXT: [0, 4] = %[[SIZE]],
          // CHECK-NEXT: [1, 2] = %[[SIZE]]
          [1, 2] = %size,
          [0, 4] = %size,
        }) : index
  // CHECK: util.return %[[PACKED]]#0, %[[PACKED]]#2, %[[PACKED]]#1
  util.return %total_length, %offset_0, %offset_1 : index, index, index
}

// -----

// CHECK-LABEL: @FoldResourceSubviewOp
util.func private @FoldResourceSubviewOp(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.resource.subview
  %0 = stream.resource.subview %arg0[%c0] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
  // CHECK: util.return %arg0
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @FoldResourceSubviewOps
util.func private @FoldResourceSubviewOps(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %c300 = arith.constant 300 : index
  %c400 = arith.constant 400 : index
  %c500 = arith.constant 500 : index
  // CHECK: %[[RET:.+]] = stream.resource.subview %arg0[%c300] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c300}
  %0 = stream.resource.subview %arg0[%c100] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c500}
  %1 = stream.resource.subview %0[%c100] : !stream.resource<*>{%c500} -> !stream.resource<*>{%c400}
  %2 = stream.resource.subview %1[%c100] : !stream.resource<*>{%c400} -> !stream.resource<*>{%c300}
  // CHECK-NEXT: util.return %[[RET]]
  util.return %2 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @SinkSubviewAcrossSelectOps
util.func private @SinkSubviewAcrossSelectOps(%arg0: !stream.resource<*>, %arg1: i1) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK-NOT: stream.resource.subview
  %0 = stream.resource.subview %arg0[%c0] : !stream.resource<*>{%c256} -> !stream.resource<*>{%c128}
  // CHECK-NOT: stream.resource.subview
  %1 = stream.resource.subview %arg0[%c128] : !stream.resource<*>{%c256} -> !stream.resource<*>{%c128}
  // CHECK: %[[OFFSET:.+]] = arith.select %arg1, %c0, %c128 : index
  %2 = arith.select %arg1, %0, %1 : !stream.resource<*>
  // CHECK-NEXT: %[[SUBVIEW:.+]] = stream.resource.subview %arg0[%[[OFFSET]]] : !stream.resource<*>{%c256} -> !stream.resource<*>{%c128}
  // CHECK-NEXT: util.return %[[SUBVIEW]]
  util.return %2 : !stream.resource<*>
}

// -----

// Tests that unrealized_conversion_casts on resources are properly cleaned up.

// CHECK-LABEL: unrealizedCastCleanup
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<transient>, %[[ARG1:.+]]: index)
util.func private @unrealizedCastCleanup(%arg0: !stream.resource<transient>, %arg1: index) -> (!stream.resource<transient>, index) {
  %0 = builtin.unrealized_conversion_cast %arg0, %arg1 : !stream.resource<transient>, index to !stream.resource<transient>
  %1 = stream.resource.size %0 : !stream.resource<transient>
  // CHECK-NEXT: util.return %[[ARG0]], %[[ARG1]]
  util.return %0, %1 : !stream.resource<transient>, index
}
