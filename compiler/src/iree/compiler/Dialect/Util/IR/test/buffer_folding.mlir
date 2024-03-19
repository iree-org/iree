// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldSubspansIntoSliceOp
util.func public @FoldSubspansIntoSliceOp(%arg0: !util.buffer, %arg1: index, %arg2: index, %arg3: index) -> !util.buffer {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[OFFSET:.+]] = arith.addi %arg2, %c100
  %0 = util.buffer.subspan %arg0[%arg2] : !util.buffer{%arg1} -> !util.buffer{%arg3}
  // CHECK: util.buffer.slice %arg0[%[[OFFSET]]] : !util.buffer{%arg1} -> !util.buffer{%c200}
  %1 = util.buffer.slice %0[%c100] : !util.buffer{%arg3} -> !util.buffer{%c200}
  util.return %1 : !util.buffer
}

// -----

// CHECK-LABEL: @FoldBufferSubspanOp
util.func public @FoldBufferSubspanOp(%arg0: !util.buffer, %arg1: index, %arg2: index) -> !util.buffer {
  // CHECK-NOT: util.buffer.subspan
  %0 = util.buffer.subspan %arg0[%arg1] : !util.buffer{%arg2} -> !util.buffer{%arg2}
  // CHECK: util.return %arg0
  util.return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @FoldBufferSubspanOps
util.func public @FoldBufferSubspanOps(%arg0: !util.buffer, %arg1: index) -> !util.buffer {
  %c100 = arith.constant 100 : index
  %c300 = arith.constant 300 : index
  %c400 = arith.constant 400 : index
  %c500 = arith.constant 500 : index
  // CHECK: %[[RET:.+]] = util.buffer.subspan %arg0[%c300] : !util.buffer{%arg1} -> !util.buffer{%c300}
  %0 = util.buffer.subspan %arg0[%c100] : !util.buffer{%arg1} -> !util.buffer{%c500}
  %1 = util.buffer.subspan %0[%c100] : !util.buffer{%c500} -> !util.buffer{%c400}
  %2 = util.buffer.subspan %1[%c100] : !util.buffer{%c400} -> !util.buffer{%c300}
  // CHECK: util.return %[[RET]]
  util.return %2 : !util.buffer
}

// -----

// CHECK-LABEL: @SinkSubspanAcrossSelectOps
util.func public @SinkSubspanAcrossSelectOps(%arg0: !util.buffer, %arg1: i1) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK-NOT: util.buffer.subspan
  %0 = util.buffer.subspan %arg0[%c0] : !util.buffer{%c256} -> !util.buffer{%c128}
  // CHECK-NOT: util.buffer.subspan
  %1 = util.buffer.subspan %arg0[%c128] : !util.buffer{%c256} -> !util.buffer{%c128}
  // CHECK: %[[OFFSET:.+]] = arith.select %arg1, %c0, %c128 : index
  %2 = arith.select %arg1, %0, %1 : !util.buffer
  // CHECK-NEXT: %[[SUBSPAN:.+]] = util.buffer.subspan %arg0[%[[OFFSET]]] : !util.buffer{%c256} -> !util.buffer{%c128}
  // CHECK-NEXT: util.return %[[SUBSPAN]]
  util.return %2 : !util.buffer
}

// -----

// CHECK-LABEL: @FoldBufferSizeOp
util.func public @FoldBufferSizeOp(%arg0: !util.buffer, %arg1: index) -> (index, i32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // CHECK-NOT: util.buffer.size
  %0 = util.buffer.size %arg0 : !util.buffer
  // CHECK: %[[LOAD:.+]] = util.buffer.load
  %1 = util.buffer.load %arg0[%c0 for %c4] : !util.buffer{%arg1} -> i32
  // CHECK: util.return %arg1, %[[LOAD]]
  util.return %0, %1 : index, i32
}

// -----

// CHECK-LABEL: @FoldNestedBufferSizeOp
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @FoldNestedBufferSizeOp(%buffer: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  // CHECK: scf.for
  scf.for %i = %c0 to %c128 step %c1 {
    // CHECK: %[[BUFFER_SIZE_INNER:.+]] = util.buffer.size %[[BUFFER]]
    %buffer_size_inner = util.buffer.size %buffer : !util.buffer
    // CHECK: util.buffer.load %[[BUFFER]]{{.+}} : !util.buffer{%[[BUFFER_SIZE_INNER]]}
    %inner = util.buffer.load %buffer[%i for %c1] : !util.buffer{%buffer_size_inner} -> i8
    util.optimization_barrier %inner : i8
  }
  // CHECK: %[[BUFFER_SIZE_OUTER:.+]] = util.buffer.size %[[BUFFER]]
  %buffer_size_outer = util.buffer.size %buffer : !util.buffer
  // CHECK: util.buffer.load %[[BUFFER]]{{.+}} : !util.buffer{%[[BUFFER_SIZE_OUTER]]}
  %outer = util.buffer.load %buffer[%c128 for %c1] : !util.buffer{%buffer_size_outer} -> i8
  util.optimization_barrier %outer : i8
  util.return
}

// -----

// CHECK-LABEL: @FoldConstantBufferSizeOp
util.func public @FoldConstantBufferSizeOp() -> index {
  // CHECK-NOT: util.buffer.constant
  %0 = util.buffer.constant : !util.buffer = dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK-NOT: util.buffer.size
  %1 = util.buffer.size %0 : !util.buffer
  // CHECK: util.return %c12
  util.return %1 : index
}

// -----

// CHECK-LABEL: @SelectBufferSizeOp
util.func public @SelectBufferSizeOp(%arg0: !util.buffer, %arg1: index, %arg2: !util.buffer, %arg3: index, %arg4: i1) -> (!util.buffer, index) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[ARG0_T:.+]] = util.buffer.slice %arg0[%c0] : !util.buffer{%[[ARG0_SZ:.+]]} ->
  %0 = util.buffer.slice %arg0[%c0] : !util.buffer{%arg1} -> !util.buffer{%arg1}
  // CHECK: %[[ARG2_T:.+]] = util.buffer.slice %arg2[%c0] : !util.buffer{%[[ARG2_SZ:.+]]} ->
  %1 = util.buffer.slice %arg2[%c0] : !util.buffer{%arg3} -> !util.buffer{%arg3}
  // CHECK: %[[RET_T:.+]] = arith.select %arg4, %[[ARG0_T]], %[[ARG2_T]] : !util.buffer
  %2 = arith.select %arg4, %0, %1 : !util.buffer
  // CHECK: %[[RET_SIZE:.+]] = arith.select %arg4, %[[ARG0_SZ]], %[[ARG2_SZ]] : index
  %3 = util.buffer.size %2 : !util.buffer
  // CHECK: = util.buffer.slice %[[RET_T]][%c0] : !util.buffer{%[[RET_SIZE]]} ->
  %4 = util.buffer.slice %2[%c0] : !util.buffer{%3} -> !util.buffer{%3}
  util.return %4, %3 : !util.buffer, index
}

// -----

// CHECK-LABEL: @FoldSubspansIntoStorageOp
util.func public @FoldSubspansIntoStorageOp(%arg0: !util.buffer, %arg1: index, %arg2: index, %arg3: index) -> (memref<?xi8>, index) {
  // CHECK-NOT: util.buffer.subspan
  %0 = util.buffer.subspan %arg0[%arg2] : !util.buffer{%arg1} -> !util.buffer{%arg3}
  // CHECK: %[[STORAGE:.+]], %[[OFFSET:.+]] = util.buffer.storage %arg0 : !util.buffer{%arg1} -> (memref<?xi8>, index)
  %1:2 = util.buffer.storage %0 : !util.buffer{%arg3} -> (memref<?xi8>, index)
  // CHECK: %[[ADJUSTED_OFFSET:.+]] = arith.addi %arg2, %[[OFFSET]]
  // CHECK: util.return %[[STORAGE]], %[[ADJUSTED_OFFSET]]
  util.return %1#0, %1#1 : memref<?xi8>, index
}

// -----

// CHECK-LABEL: @FoldSubspansIntoCopyOp
util.func public @FoldSubspansIntoCopyOp(%arg0: !util.buffer, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[OFFSET_SRC:.+]] = arith.addi %arg2, %c100
  %0 = util.buffer.subspan %arg0[%arg2] : !util.buffer{%arg1} -> !util.buffer{%arg3}
  // CHECK: %[[OFFSET_DST:.+]] = arith.addi %arg4, %c200
  %1 = util.buffer.subspan %arg0[%arg4] : !util.buffer{%arg1} -> !util.buffer{%arg5}
  // CHECK: util.buffer.copy %arg0[%[[OFFSET_SRC]]], %arg0[%[[OFFSET_DST]]], %c1 : !util.buffer{%arg1} -> !util.buffer{%arg1}
  util.buffer.copy %0[%c100], %1[%c200], %c1 : !util.buffer{%arg3} -> !util.buffer{%arg5}
  util.return
}

// -----

// CHECK-LABEL: @FoldSubspansIntoCompareOp
util.func public @FoldSubspansIntoCompareOp(%arg0: !util.buffer, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> i1 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[OFFSET_LHS:.+]] = arith.addi %arg2, %c100
  %0 = util.buffer.subspan %arg0[%arg2] : !util.buffer{%arg1} -> !util.buffer{%arg3}
  // CHECK: %[[OFFSET_RHS:.+]] = arith.addi %arg4, %c200
  %1 = util.buffer.subspan %arg0[%arg4] : !util.buffer{%arg1} -> !util.buffer{%arg5}
  // CHECK: = util.buffer.compare %arg0[%[[OFFSET_LHS]]], %arg0[%[[OFFSET_RHS]]], %c1 : !util.buffer{%arg1}, !util.buffer{%arg1}
  %2 = util.buffer.compare %0[%c100], %1[%c200], %c1 : !util.buffer{%arg3}, !util.buffer{%arg5}
  util.return %2 : i1
}

// -----

// CHECK-LABEL: @FoldSubspansIntoFillOp
util.func public @FoldSubspansIntoFillOp(%arg0: !util.buffer, %arg1: index, %arg2: i32, %arg3: index, %arg4: index) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[OFFSET:.+]] = arith.addi %arg3, %c100
  %0 = util.buffer.subspan %arg0[%arg3] : !util.buffer{%arg1} -> !util.buffer{%arg4}
  // CHECK: util.buffer.fill %arg2, %arg0[%[[OFFSET]] for %c200] : i32 -> !util.buffer{%arg1}
  util.buffer.fill %arg2, %0[%c100 for %c200] : i32 -> !util.buffer{%arg4}
  util.return
}

// -----

// CHECK-LABEL: @FoldSubspanIntoLoadOp
util.func public @FoldSubspanIntoLoadOp(%arg0: !util.buffer, %arg1: index) -> i32 {
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK-NOT: util.buffer.subspan
  %0 = util.buffer.subspan %arg0[%c128] : !util.buffer{%arg1} -> !util.buffer{%c256}
  // CHECK: = util.buffer.load %arg0[%c192 for %c4] : !util.buffer{%arg1} -> i32
  %1 = util.buffer.load %0[%c64 for %c4] : !util.buffer{%c256} -> i32
  util.return %1 : i32
}

// -----

// CHECK-LABEL: @FoldSubspanIntoStoreOp
util.func public @FoldSubspanIntoStoreOp(%arg0: !util.buffer, %arg1: index) {
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: util.buffer.subspan
  %0 = util.buffer.subspan %arg0[%c128] : !util.buffer{%arg1} -> !util.buffer{%c256}
  // CHECK: util.buffer.store %c123_i32, %arg0[%c192 for %c4] : i32 -> !util.buffer{%arg1}
  util.buffer.store %c123_i32, %0[%c64 for %c4] : i32 -> !util.buffer{%c256}
  util.return
}
