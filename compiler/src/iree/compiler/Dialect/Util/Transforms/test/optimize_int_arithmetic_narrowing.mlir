// RUN: iree-opt --split-input-file \
// RUN:  --iree-util-optimize-int-arithmetic=narrow-to-i32=true --cse %s \
// RUN:  | FileCheck %s
// We inherit a number of patterns from upstream for narrowing specific arith
// operations. Those are not the focus of testing, but we may test some of them
// here incidentally as part of verifying that the overall pass and local
// patterns are effective.

// CHECK-LABEL: @narrow_tid_computations
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
// CHECK-DAG: %[[THREAD_ID_X:.+]] = gpu.thread_id x upper_bound 64
// CHECK-DAG: %[[TID_I32:.+]] = arith.index_castui %[[THREAD_ID_X]] : index to i32
// CHECK: %[[V0:.+]] = arith.divui %[[TID_I32]], %[[C16]] : i32
// CHECK-NEXT: %[[V1:.+]] = arith.remui %[[TID_I32]], %[[C16]] : i32
// CHECK-NEXT: %[[V2:.+]] = arith.muli %[[V0]], %[[V32]] : i32
// CHECK-NEXT; %[[V3:.+]] = arith.addi %[[V2]], %[[V1]] : i32
// CHECK-NEXT: %[[RET:.+]] = arith.index_castui %[[V3]] : i32 to index
// CHECK: return %[[RET]]
util.func @narrow_tid_computations() -> index {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %thread_id_x = gpu.thread_id x upper_bound 64
  %0 = arith.divui %thread_id_x, %c16 : index
  %1 = arith.remui %thread_id_x, %c16 : index
  %2 = arith.muli %0, %c32 : index
  %3 = arith.addi %2, %1 : index
  util.return %3 : index
}

// -----

// CHECK-LABEL: @narrow_assumes
// CHECK-SAME: (%[[ARG0:.+]]: i32)
// CHECK-NEXT: %[[ASSUME:.+]] = util.assume.int %[[ARG0]][<umin = 16, umax = 122, udiv = 16>] : i32
// CHECK-NEXT: %[[AS_INDEX:.+]] = arith.index_castui %[[ASSUME]] : i32 to index
// CHECK-NEXT: util.return %[[ASSUME]], %[[AS_INDEX]]
util.func @narrow_assumes(%arg0: i32) -> (i32, index) {
  %0 = arith.index_castui %arg0 : i32 to index
  %1 = util.assume.int %0[<umin = 16, umax = 122, udiv = 16>] : index
  %2 = arith.index_castui %1 : index to i32
  util.return %2, %1 : i32, index
}
