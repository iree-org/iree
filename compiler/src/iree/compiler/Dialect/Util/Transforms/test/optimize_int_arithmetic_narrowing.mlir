// RUN: iree-opt --split-input-file \
// RUN:  --iree-util-optimize-int-arithmetic=narrow-to-i32=true --cse %s \
// RUN:  | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: iree-opt --split-input-file \
// RUN:  --iree-util-optimize-int-arithmetic="narrow-to-i32=true index-is-i64=true" --cse %s \
// RUN:  | FileCheck %s --check-prefixes=CHECK,I64
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

// -----

// Verify that broadcast(index_castui(x)) is rewritten to index_castui(broadcast(x)).
// CHECK-LABEL: @narrow_broadcast_index_castui
// CHECK-SAME: (%[[ARG0:.+]]: i32)
// CHECK: %[[BCAST:.+]] = vector.broadcast %[[ARG0]] : i32 to vector<4xi32>
// CHECK: %[[CAST:.+]] = arith.index_castui %[[BCAST]] : vector<4xi32> to vector<4xindex>
// CHECK: return %[[CAST]]
func.func @narrow_broadcast_index_castui(%arg0: i32) -> vector<4xindex> {
  %cast = arith.index_castui %arg0 : i32 to index
  %bcast = vector.broadcast %cast : index to vector<4xindex>
  return %bcast : vector<4xindex>
}

// -----

// Verify that we take divsi to divui on index if the input is non-negative as
// an i64 but out of range as an i32 only if we know that index is i64.
// CHECK-LABEL: @index_divsi_beyond_i32_range
// DEFAULT: arith.divsi
// I64: arith.divui
util.func @index_divsi_beyond_i32_range(%arg0 : index) -> index {
  %cst = arith.constant 5 : index
  %0 = util.assume.int %arg0<umin=10, umax=5000000000> : index
  %1 = arith.divsi %0, %cst : index
  util.return %1 : index
}

// -----

// Verify that broadcast(index_cast(x)) is rewritten to index_cast(broadcast(x)).
// CHECK-LABEL: @narrow_broadcast_index_cast
// CHECK-SAME: (%[[ARG0:.+]]: i32)
// CHECK: %[[BCAST:.+]] = vector.broadcast %[[ARG0]] : i32 to vector<2xi32>
// CHECK: %[[CAST:.+]] = arith.index_cast %[[BCAST]] : vector<2xi32> to vector<2xindex>
// CHECK: return %[[CAST]]
func.func @narrow_broadcast_index_cast(%arg0: i32) -> vector<2xindex> {
  %cast = arith.index_cast %arg0 : i32 to index
  %bcast = vector.broadcast %cast : index to vector<2xindex>
  return %bcast : vector<2xindex>
}

// -----

// Verify that we move index casts across arithmetic if the inputs are outside
// of i32 range only if we know that index is i64.
// CHECK-LABEL: @i64_index_cast_beyond_i32_range
// DEFAULT: arith.addi %{{.*}}, %{{.*}} : i64
// I64: arith.addi %{{.*}}, %{{.*}} : index
util.func @i64_index_cast_beyond_i32_range(%arg0 : i64) -> index {
  %c1 = arith.constant 1 : i64
  %0 = util.assume.int %arg0<umin=10, umax=4294967295> : i64
  %1 = arith.addi %0, %c1 : i64
  %2 = arith.index_castui %1 : i64 to index
  util.return %2 : index
}

// -----

// Verify that broadcast(index_castui(vector)) is also rewritten.
// CHECK-LABEL: @narrow_broadcast_vector_index_castui
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>)
// CHECK: %[[BCAST:.+]] = vector.broadcast %[[ARG0]] : vector<4xi32> to vector<2x4xi32>
// CHECK: %[[CAST:.+]] = arith.index_castui %[[BCAST]] : vector<2x4xi32> to vector<2x4xindex>
// CHECK: return %[[CAST]]
func.func @narrow_broadcast_vector_index_castui(%arg0: vector<4xi32>) -> vector<2x4xindex> {
  %cast = arith.index_castui %arg0 : vector<4xi32> to vector<4xindex>
  %bcast = vector.broadcast %cast : vector<4xindex> to vector<2x4xindex>
  return %bcast : vector<2x4xindex>
}

// -----

// Negative test: broadcast is on i32 (not index) — should NOT be rewritten.
// CHECK-LABEL: @no_narrow_broadcast_non_index
// CHECK: %[[CAST:.+]] = arith.index_cast %{{.*}} : index to i32
// CHECK: %[[BCAST:.+]] = vector.broadcast %[[CAST]] : i32 to vector<4xi32>
// CHECK: return %[[BCAST]]
func.func @no_narrow_broadcast_non_index(%arg0: index) -> vector<4xi32> {
  %cast = arith.index_cast %arg0 : index to i32
  %bcast = vector.broadcast %cast : i32 to vector<4xi32>
  return %bcast : vector<4xi32>
}

// -----

// Verify that we only move index_casts across i64 orithmetic if we know the
// inputs are i64.
// CHECK-LABEL: @i64_index_cast_no_range_info
// DEFAULT: arith.addi %{{.*}}, %{{.*}} : i64
// I64: arith.addi %{{.*}}, %{{.*}} : index
util.func @i64_index_cast_no_range_info(%arg0 : i64, %arg1 : i64) -> index {
  %0 = arith.addi %arg0, %arg1 : i64
  %1 = arith.index_castui %0 : i64 to index
  util.return %1 : index
}

// -----

// Negative test: ensure we don't narrow scf.for IVs to i32 when they might
// be out of i32 range.
// CHECK-LABEL: @no_narrow_scf_for_large_iv
// CHECK-NOT: : i32
// CHECK: util.return
util.func @no_narrow_scf_for_large_iv(%arg0: memref<?xf32>) {
  %c0_f32 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3000000000 = arith.constant 3000000000 : index
  scf.for %iv = %c0 to %c3000000000 step %c1 {
    memref.store %c0_f32, %arg0[%iv] : memref<?xf32>
  }
  util.return
}
