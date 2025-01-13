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
// CHECK-NEXT: %[[V2:.+]] = arith.muli %[[V0]], %[[C32]] : i32
// CHECK-NEXT: %[[V3:.+]] = arith.addi %[[V2]], %[[V1]] : i32
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
// CHECK-NEXT: %[[ASSUME:.+]] = util.assume.int %[[ARG0]]<umin = 16, umax = 122, udiv = 16> : i32
// CHECK-NEXT: %[[AS_INDEX:.+]] = arith.index_castui %[[ASSUME]] : i32 to index
// CHECK-NEXT: util.return %[[ASSUME]], %[[AS_INDEX]]
util.func @narrow_assumes(%arg0: i32) -> (i32, index) {
  %0 = arith.index_castui %arg0 : i32 to index
  %1 = util.assume.int %0<umin = 16, umax = 122, udiv = 16> : index
  %2 = arith.index_castui %1 : index to i32
  util.return %2, %1 : i32, index
}

// -----

// CHECK-LABEL: @narrow_scf_for
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : i32
// CHECK-DAG: %[[C96:.+]] = arith.constant 96 : i32
// CHECK-DAG: %[[C512:.+]] = arith.constant 512 : i32
// CHECK-DAG: %[[TID:.+]] = gpu.thread_id x upper_bound 64
// CHECK-DAG: %[[TID_I32:.+]] = arith.index_castui %[[TID]] : index to i32
// CHECK: scf.for %[[ARG1:.+]] = %[[TID_I32]] to %[[C96]] step %[[C64]]
// CHECK-NEXT: %[[V0:.+]] = arith.addi %[[ARG1]], %[[C512]]
// CHECK-NEXT: %[[V0_IDX:.+]] = arith.index_castui %[[V0]] : i32 to index
// CHECK-NEXT: memref.store {{.*}}[%[[V0_IDX]]]
util.func @narrow_scf_for(%arg0: memref<?xf32>) {
  %c0_f32 = arith.constant 0.0 : f32
  %c64 = arith.constant 64 : index
  %c96 = arith.constant 96 : index
  %c512 = arith.constant 512 : index
  %tid = gpu.thread_id x upper_bound 64
  scf.for %arg1 = %tid to %c96 step %c64 {
    %0 = arith.addi %arg1, %c512 : index
    memref.store %c0_f32, %arg0[%0] : memref<?xf32>
  }
  util.return
}
