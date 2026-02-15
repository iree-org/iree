// RUN: iree-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-canonicalize-ids, canonicalize, cse))" | \
// RUN:   FileCheck %s

// Test: gpu.subgroup_size folds to constant.
func.func @fold_subgroup_size() -> index
    attributes {
      translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
    } {
  %0 = gpu.subgroup_size : index
  return %0 : index
}

// CHECK-LABEL: func.func @fold_subgroup_size
//       CHECK:   %[[C64:.+]] = arith.constant 64 : index
//       CHECK:   return %[[C64]]

// -----

// Test: gpu.block_dim x/y/z fold to constants.
func.func @fold_block_dim() -> (index, index, index)
    attributes {
      translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
    } {
  %0 = gpu.block_dim x
  %1 = gpu.block_dim y
  %2 = gpu.block_dim z
  return %0, %1, %2 : index, index, index
}

// CHECK-LABEL: func.func @fold_block_dim
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   return %[[C128]], %[[C2]], %[[C1]]

// -----

// Test: gpu.thread_id z folds to 0 when z-dim is 1.
func.func @fold_thread_id_to_zero() -> (index, index, index)
    attributes {
      translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
    } {
  %0 = gpu.thread_id x
  %1 = gpu.thread_id y
  %2 = gpu.thread_id z
  return %0, %1, %2 : index, index, index
}

// CHECK-LABEL: func.func @fold_thread_id_to_zero
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[TX:.+]] = gpu.thread_id x
//       CHECK:   %[[TY:.+]] = gpu.thread_id y
//       CHECK:   return %[[TX]], %[[TY]], %[[C0]]

// -----

// Test: gpu.subgroup_id rewrites to linearized thread ID / subgroup_size.
func.func @rewrite_subgroup_id() -> index
    attributes {
      translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
    } {
  %0 = gpu.subgroup_id : index
  return %0 : index
}

// The linearization is: tid.x + 128 * tid.y (z-dim is 1, folded away).
// Then divui by 64.
// CHECK-LABEL: func.func @rewrite_subgroup_id
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//       CHECK:   %[[MUL:.+]] = arith.muli %[[TY]], %[[C128]]
//       CHECK:   %[[LINEAR:.+]] = arith.addi %[[TX]], %[[MUL]]
//       CHECK:   %[[RESULT:.+]] = arith.divui %[[LINEAR]], %[[C64]]
//       CHECK:   return %[[RESULT]]

// -----

// Test: gpu.lane_id rewrites to linearized thread ID % subgroup_size.
func.func @rewrite_lane_id() -> index
    attributes {
      translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
    } {
  %0 = gpu.lane_id
  return %0 : index
}

// CHECK-LABEL: func.func @rewrite_lane_id
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//       CHECK:   %[[MUL:.+]] = arith.muli %[[TY]], %[[C128]]
//       CHECK:   %[[LINEAR:.+]] = arith.addi %[[TX]], %[[MUL]]
//       CHECK:   %[[RESULT:.+]] = arith.remui %[[LINEAR]], %[[C64]]
//       CHECK:   return %[[RESULT]]

// -----

// Test: gpu.num_subgroups folds to constant (128*2*1)/64 = 4.
func.func @fold_num_subgroups() -> index
    attributes {
      translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
    } {
  %0 = gpu.num_subgroups : index
  return %0 : index
}

// CHECK-LABEL: func.func @fold_num_subgroups
//       CHECK:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   return %[[C4]]

// -----

// Test: 1D workgroup - subgroup_id simplifies to tid.x / 64.
func.func @subgroup_id_1d() -> index
    attributes {
      translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [512, 1, 1] subgroup_size = 64>
    } {
  %0 = gpu.subgroup_id : index
  return %0 : index
}

// With [512, 1, 1], y and z are folded to 0, so linear = tid.x.
// CHECK-LABEL: func.func @subgroup_id_1d
//       CHECK:   %[[C64:.+]] = arith.constant 64 : index
//       CHECK:   %[[TX:.+]] = gpu.thread_id x
//       CHECK:   %[[RESULT:.+]] = arith.divui %[[TX]], %[[C64]]
//       CHECK:   return %[[RESULT]]

// -----

// Test: No translation info - pass is a no-op.
func.func @no_translation_info() -> index {
  %0 = gpu.subgroup_size : index
  return %0 : index
}

// CHECK-LABEL: func.func @no_translation_info
//       CHECK:   %[[SZ:.+]] = gpu.subgroup_size : index
//       CHECK:   return %[[SZ]]
