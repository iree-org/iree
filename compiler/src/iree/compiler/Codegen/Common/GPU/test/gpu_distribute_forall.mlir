// RUN: iree-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-distribute-forall, canonicalize, cse))" | FileCheck %s

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_thread_forall(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0) in (1024) {
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_thread_forall
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[TZ:.+]] = gpu.thread_id z
//       CHECK:   scf.for %[[I:.+]] = %c0 to %c1024 step %c128 {
//       CHECK:     %[[LINID:.+]] = affine.apply
//  CHECK-SAME:       affine_map<(d0)[s0, s1, s2] -> (d0 + s0 + s1 * 64 + s2 * 128)>(%[[I]])
//  CHECK-SAME:       [%[[TX]], %[[TY]], %[[TZ]]]
//       CHECK:     %[[DELIN:.+]] = affine.delinearize_index %[[LINID]] into (%c1024) : index
//       CHECK:     memref.store {{.*}}[%[[DELIN]]]

// -----

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_warp_forall(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0) in (32) {
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_warp_forall
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[TZ:.+]] = gpu.thread_id z
//       CHECK:   scf.for %[[I:.+]] = %c0 to %c32 step %c4 {
//       CHECK:     %[[LINID:.+]] = affine.apply
//  CHECK-SAME:       affine_map<(d0)[s0, s1, s2] -> (d0 + s1 * 2 + s2 * 4 + s0 floordiv 32)>(%[[I]])
//  CHECK-SAME:       [%[[TX]], %[[TY]], %[[TZ]]]
//       CHECK:     %[[DELIN:.+]] = affine.delinearize_index %[[LINID]] into (%c32) : index
//       CHECK:     memref.store {{.*}}[%[[DELIN]]]

// -----

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_lane_forall(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  scf.forall (%arg0) in (32) {
    %c0 = arith.constant 0 : i32
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#iree_gpu.lane_id<0>]}
  return
}

// CHECK-LABEL: func @distribute_lane_forall
//       CHECK:   %[[LANEID:.+]] = gpu.lane_id
//       CHECK:   memref.store {{.*}}[%[[LANEID]]]

// -----

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_thread_forall_drop_for_loop(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0) in (128) {
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_thread_forall_drop_for_loop
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[TZ:.+]] = gpu.thread_id z
//   CHECK-NOT:   scf.for
//       CHECK:   %[[LINID:.+]] = affine.apply
//  CHECK-SAME:     affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 128)>
//  CHECK-SAME:     [%[[TX]], %[[TY]], %[[TZ]]]
//       CHECK:   %[[DELIN:.+]] = affine.delinearize_index %[[LINID]] into (%c128) : index
//       CHECK:   memref.store {{.*}}[%[[DELIN]]]

// -----

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_thread_forall_single_thread(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0) in (1) {
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_thread_forall_single_thread
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[TZ:.+]] = gpu.thread_id z
//       CHECK:   %[[LINID:.+]] = affine.apply
//  CHECK-SAME:     affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 128)>
//  CHECK-SAME:     [%[[TX]], %[[TY]], %[[TZ]]]
//       CHECK:   scf.for %[[I:.+]] = %[[LINID]] to %c1 step %c128 {
//       CHECK:     memref.store {{.*}}[%[[C0]]]

// -----

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_thread_forall_multi_dim(%out : memref<?x?x?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0, %arg1, %arg2) in (16, 8, 4) {
    memref.store %c0, %out[%arg0, %arg1, %arg2] : memref<?x?x?xi32>
  } {mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_thread_forall_multi_dim
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[TZ:.+]] = gpu.thread_id z
//       CHECK:   scf.for %[[I:.+]] = %c0 to %c512 step %c128 {
//       CHECK:     %[[LINID:.+]] = affine.apply
//  CHECK-SAME:       affine_map<(d0)[s0, s1, s2] -> (d0 + s0 + s1 * 64 + s2 * 128)>(%[[I]])
//  CHECK-SAME:       [%[[TX]], %[[TY]], %[[TZ]]]
//       CHECK:     %[[DELIN:.+]]:3 = affine.delinearize_index %[[LINID]] into (%c16, %c8, %c4) : index
//       CHECK:     memref.store {{.*}}[%[[DELIN]]#0, %[[DELIN]]#1, %[[DELIN]]#2]


// -----

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [7, 1, 1] subgroup_size = 32>

func.func @distribute_thread_forall_small_workgroup(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0) in (7) {
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_thread_forall_small_workgroup
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[TZ:.+]] = gpu.thread_id z
//       CHECK:   %[[LINID:.+]] = affine.apply
//  CHECK-SAME:     affine_map<()[s0, s1, s2] -> (s0 + s1 * 7 + s2 * 7)>
//  CHECK-SAME:     [%[[TX]], %[[TY]], %[[TZ]]]
//       CHECK:   %[[DELIN:.+]] = affine.delinearize_index %[[LINID]] into (%c7) : index
//       CHECK:   memref.store {{.*}}[%[[DELIN]]]
