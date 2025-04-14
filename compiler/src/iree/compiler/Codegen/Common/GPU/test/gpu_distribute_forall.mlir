// RUN: iree-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-distribute-forall, canonicalize, cse))" | FileCheck %s

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

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
//       CHECK:   %[[TFLAT:.+]] = affine.linearize_index disjoint [%[[TY]], %[[TX]]] by (2, 64)
//       CHECK:   gpu.barrier
//       CHECK:   scf.for %[[I:.+]] = %c0 to %c1024 step %c128 {
//       CHECK:     %[[LINID:.+]] = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%[[I]])[%[[TFLAT]]]
//       CHECK:     memref.store {{.*}}[%[[LINID]]]
//       CHECK:   gpu.barrier

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

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
//       CHECK:   %[[TFLAT:.+]] = affine.linearize_index disjoint [%[[TY]], %[[TX]]] by (2, 64)
//       CHECK:   %[[WARPSPLIT:.+]]:2 = affine.delinearize_index %[[TFLAT]] into (4, 32)
//       CHECK:   gpu.barrier
//       CHECK:   scf.for %[[I:.+]] = %c0 to %c32 step %c4 {
//       CHECK:     %[[LINID:.+]] = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%[[I]])[%[[WARPSPLIT]]#0]
//       CHECK:     memref.store {{.*}}[%[[LINID]]]
//       CHECK:   gpu.barrier

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_lane_forall(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  scf.forall (%arg0) in (32) {
    %c0 = arith.constant 0 : i32
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#iree_gpu.lane_id<0>]}
  return
}

// CHECK-LABEL: func @distribute_lane_forall
//       CHECK:   %[[LANEID:.+]] = gpu.lane_id upper_bound 32
//       CHECK:   memref.store {{.*}}[%[[LANEID]]]

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

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
//       CHECK:   %[[LINID:.+]] = affine.linearize_index disjoint [%[[TY]], %[[TX]]] by (2, 64)
//       CHECK:   gpu.barrier
//       CHECK:   memref.store {{.*}}[%[[LINID]]]
//       CHECK:   gpu.barrier

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_thread_forall_single_thread(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0) in (1) {
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_thread_forall_single_thread
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//       CHECK:   %[[TFLAT:.+]] = affine.linearize_index disjoint [%[[TY]], %[[TX]]] by (2, 64)
//       CHECK:   gpu.barrier
//       CHECK:   scf.for %[[I:.+]] = %[[TFLAT]] to %c1 step %c128 {
//       CHECK:     memref.store {{.*}}[%[[I]]]
//       CHECK:   gpu.barrier

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

func.func @distribute_thread_forall_overhang(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0) in (513) {
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_thread_forall_overhang
//   CHECK-DAG:   %[[C513:.+]] = arith.constant 513 : index
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//       CHECK:   %[[TFLAT:.+]] = affine.linearize_index disjoint [%[[TY]], %[[TX]]] by (2, 64)
//       CHECK:   gpu.barrier
//       CHECK:   scf.for %[[I:.+]] = %[[TFLAT]] to %[[C513]] step %c128 {
//       CHECK:     memref.store {{.*}}[%[[I]]]
//       CHECK:   gpu.barrier

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

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
//       CHECK:   %[[TFLAT:.+]] = affine.linearize_index disjoint [%[[TY]], %[[TX]]] by (2, 64)
//       CHECK:   gpu.barrier
//       CHECK:   scf.for %[[I:.+]] = %c0 to %c512 step %c128 {
//       CHECK:     %[[LINID:.+]] = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%[[I]])[%[[TFLAT]]]
//       CHECK:     %[[DELIN:.+]]:3 = affine.delinearize_index %[[LINID]] into (16, 8, 4) : index
//       CHECK:     memref.store {{.*}}[%[[DELIN]]#0, %[[DELIN]]#1, %[[DELIN]]#2]
//       CHECK:   gpu.barrier

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [7, 1, 1] subgroup_size = 32>

func.func @distribute_thread_forall_small_workgroup(%out : memref<?xi32>)
    attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : i32
  scf.forall (%arg0) in (7) {
    memref.store %c0, %out[%arg0] : memref<?xi32>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func @distribute_thread_forall_small_workgroup
//       CHECK:   %[[TX:.+]] = gpu.thread_id x
//       CHECK:   gpu.barrier
//       CHECK:   memref.store {{.*}}[%[[TX]]]
//       CHECK:   gpu.barrier
