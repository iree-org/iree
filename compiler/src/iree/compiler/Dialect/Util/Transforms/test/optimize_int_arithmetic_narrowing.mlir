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

// Narrow select(cond, index_cast(x), index_cast(y)) to operate in i32.
// CHECK-LABEL: @narrow_select_both_casts
// CHECK-SAME: (%[[COND:.+]]: i1, %[[A:.+]]: i32, %[[B:.+]]: i32)
// CHECK: %[[SEL:.+]] = arith.select %[[COND]], %[[A]], %[[B]] : i32
// CHECK: %[[CAST:.+]] = arith.index_cast %[[SEL]] : i32 to index
// CHECK: return %[[CAST]]
func.func @narrow_select_both_casts(%cond: i1, %a: i32, %b: i32) -> index {
  %ca = arith.index_cast %a : i32 to index
  %cb = arith.index_cast %b : i32 to index
  %sel = arith.select %cond, %ca, %cb : index
  return %sel : index
}

// -----

// Narrow select(cond, index_cast(x), constant 0) to operate in i32.
// CHECK-LABEL: @narrow_select_cast_and_constant
// CHECK-SAME: (%[[COND:.+]]: i1, %[[A:.+]]: i32)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK: %[[SEL:.+]] = arith.select %[[COND]], %[[A]], %[[C0]] : i32
// CHECK: %[[CAST:.+]] = arith.index_cast %[[SEL]] : i32 to index
// CHECK: return %[[CAST]]
func.func @narrow_select_cast_and_constant(%cond: i1, %a: i32) -> index {
  %ca = arith.index_cast %a : i32 to index
  %c0 = arith.constant 0 : index
  %sel = arith.select %cond, %ca, %c0 : index
  return %sel : index
}

// -----

// Narrow select with constant on the true side.
// CHECK-LABEL: @narrow_select_constant_and_cast
// CHECK-SAME: (%[[COND:.+]]: i1, %[[B:.+]]: i32)
// CHECK-DAG: %[[C42:.+]] = arith.constant 42 : i32
// CHECK: %[[SEL:.+]] = arith.select %[[COND]], %[[C42]], %[[B]] : i32
// CHECK: %[[CAST:.+]] = arith.index_cast %[[SEL]] : i32 to index
// CHECK: return %[[CAST]]
func.func @narrow_select_constant_and_cast(%cond: i1, %b: i32) -> index {
  %c42 = arith.constant 42 : index
  %cb = arith.index_cast %b : i32 to index
  %sel = arith.select %cond, %c42, %cb : index
  return %sel : index
}

// -----

// Negative test: select on i32 (not index) — should NOT be rewritten.
// CHECK-LABEL: @no_narrow_select_non_index
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
func.func @no_narrow_select_non_index(%cond: i1, %a: i32, %b: i32) -> i32 {
  %sel = arith.select %cond, %a, %b : i32
  return %sel : i32
}

// -----

// Negative test: different cast types — should NOT be rewritten.
// CHECK-LABEL: @no_narrow_select_mixed_casts
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
func.func @no_narrow_select_mixed_casts(%cond: i1, %a: i32, %b: i16) -> index {
  %ca = arith.index_cast %a : i32 to index
  %cb = arith.index_cast %b : i16 to index
  %sel = arith.select %cond, %ca, %cb : index
  return %sel : index
}

// -----

// Negative test: constant doesn't fit in the narrow type (i8).
// 256 is not representable as a signed i8 (-128 to 127).
// CHECK-LABEL: @no_narrow_select_constant_overflow
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
func.func @no_narrow_select_constant_overflow(%cond: i1, %a: i8) -> index {
  %ca = arith.index_cast %a : i8 to index
  %c256 = arith.constant 256 : index
  %sel = arith.select %cond, %ca, %c256 : index
  return %sel : index
}

// -----

// Negative test: operand is neither index_cast nor constant.
// CHECK-LABEL: @no_narrow_select_non_cast_operand
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
func.func @no_narrow_select_non_cast_operand(%cond: i1, %a: i32, %b: index) -> index {
  %ca = arith.index_cast %a : i32 to index
  %sel = arith.select %cond, %ca, %b : index
  return %sel : index
}
