// RUN: iree-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-all-ids-to-thread-ids, canonicalize, cse))" | \
// RUN:   FileCheck %s

// Test: gpu.subgroup_id rewrites to linearized thread ID / subgroup_size.
func.func @rewrite_subgroup_id() -> index {
  %0 = gpu.subgroup_id : index
  return %0 : index
}

// CHECK-LABEL: func.func @rewrite_subgroup_id
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[TZ:.+]] = gpu.thread_id z
//   CHECK-DAG:   %[[BDX:.+]] = gpu.block_dim x
//   CHECK-DAG:   %[[BDY:.+]] = gpu.block_dim y
//   CHECK-DAG:   %[[BDZ:.+]] = gpu.block_dim z
//       CHECK:   %[[LINEAR:.+]] = affine.linearize_index disjoint [%[[TZ]], %[[TY]], %[[TX]]] by (%[[BDZ]], %[[BDY]], %[[BDX]]) : index
//       CHECK:   %[[SG:.+]] = gpu.subgroup_size : index
//       CHECK:   %[[RESULT:.+]] = arith.divui %[[LINEAR]], %[[SG]]
//       CHECK:   return %[[RESULT]]

// -----

// Test: gpu.lane_id rewrites to linearized thread ID % subgroup_size.
func.func @rewrite_lane_id() -> index {
  %0 = gpu.lane_id
  return %0 : index
}

// CHECK-LABEL: func.func @rewrite_lane_id
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[TZ:.+]] = gpu.thread_id z
//   CHECK-DAG:   %[[BDX:.+]] = gpu.block_dim x
//   CHECK-DAG:   %[[BDY:.+]] = gpu.block_dim y
//   CHECK-DAG:   %[[BDZ:.+]] = gpu.block_dim z
//       CHECK:   %[[LINEAR:.+]] = affine.linearize_index disjoint [%[[TZ]], %[[TY]], %[[TX]]] by (%[[BDZ]], %[[BDY]], %[[BDX]]) : index
//       CHECK:   %[[SG:.+]] = gpu.subgroup_size : index
//       CHECK:   %[[RESULT:.+]] = arith.remui %[[LINEAR]], %[[SG]]
//       CHECK:   return %[[RESULT]]

// -----

// Test: Both subgroup_id and lane_id share the same linearization (CSE).
func.func @shared_linearization() -> (index, index) {
  %0 = gpu.subgroup_id : index
  %1 = gpu.lane_id
  return %0, %1 : index, index
}

// CHECK-LABEL: func.func @shared_linearization
//       CHECK:   %[[LINEAR:.+]] = affine.linearize_index disjoint
//       CHECK:   %[[SG:.+]] = gpu.subgroup_size : index
//       CHECK:   %[[DIV:.+]] = arith.divui %[[LINEAR]], %[[SG]]
//       CHECK:   %[[REM:.+]] = arith.remui %[[LINEAR]], %[[SG]]
//       CHECK:   return %[[DIV]], %[[REM]]
