// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-distribute-scf-for))" --mlir-print-local-scope %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-distribute-scf-for{use-block-dims=false}))" --mlir-print-local-scope %s | FileCheck --check-prefix=NO-BLOCK-DIM %s

#translation = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [64, 1, 1]>
func.func @distribute_to_x(%lb : index, %ub : index, %step: index, %output: memref<?xf32>)
  attributes {translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32

  %init = tensor.empty() : tensor<2x128xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.store %zero, %output[%iv] : memref<?xf32>
  } {iree.gpu.distribute_dim = 0 : index}

  return
}

// CHECK-LABEL: func.func @distribute_to_x
//  CHECK-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//   CHECK-DAG:   %[[ID:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[DIM:.+]] = gpu.block_dim x
//       CHECK:   %[[XLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       CHECK:   %[[XSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       CHECK:   scf.for %[[IV:.+]] = %[[XLB]] to %[[UB]] step %[[XSTEP]] {
//       CHECK:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>

// NO-BLOCK-DIM-LABEL: func.func @distribute_to_x
//  NO-BLOCK-DIM-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//   NO-BLOCK-DIM-DAG:   %[[ID:.+]] = gpu.thread_id x
//   NO-BLOCK-DIM-DAG:   %[[DIM:.+]] = arith.constant 64 : index
//       NO-BLOCK-DIM:   %[[XLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       NO-BLOCK-DIM:   %[[XSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       NO-BLOCK-DIM:   scf.for %[[IV:.+]] = %[[XLB]] to %[[UB]] step %[[XSTEP]] {
//       NO-BLOCK-DIM:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>


// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [1, 64, 1]>
func.func @distribute_to_y(%lb : index, %ub : index, %step: index, %output: memref<?xf32>)
  attributes {translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32

  %init = tensor.empty() : tensor<2x128xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.store %zero, %output[%iv] : memref<?xf32>
  } {iree.gpu.distribute_dim = 1 : index}

  return
}

// CHECK-LABEL: func.func @distribute_to_y
//  CHECK-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//   CHECK-DAG:   %[[ID:.+]] = gpu.thread_id y
//   CHECK-DAG:   %[[DIM:.+]] = gpu.block_dim y
//       CHECK:   %[[YLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       CHECK:   %[[YSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       CHECK:   scf.for %[[IV:.+]] = %[[YLB]] to %[[UB]] step %[[YSTEP]] {
//       CHECK:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>

// NO-BLOCK-DIM-LABEL: func.func @distribute_to_y
//  NO-BLOCK-DIM-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//   NO-BLOCK-DIM-DAG:   %[[ID:.+]] = gpu.thread_id y
//   NO-BLOCK-DIM-DAG:   %[[DIM:.+]] = arith.constant 64 : index
//       NO-BLOCK-DIM:   %[[YLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       NO-BLOCK-DIM:   %[[YSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       NO-BLOCK-DIM:   scf.for %[[IV:.+]] = %[[YLB]] to %[[UB]] step %[[YSTEP]] {
//       NO-BLOCK-DIM:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>


// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [1, 1, 64]>
func.func @distribute_to_z(%lb : index, %ub : index, %step: index, %output: memref<?xf32>)
  attributes {translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32

  %init = tensor.empty() : tensor<2x128xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.store %zero, %output[%iv] : memref<?xf32>
  } {iree.gpu.distribute_dim = 2 : index}

  return
}

// CHECK-LABEL: func.func @distribute_to_z
//  CHECK-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//   CHECK-DAG:   %[[ID:.+]] = gpu.thread_id z
//   CHECK-DAG:   %[[DIM:.+]] = gpu.block_dim z
//       CHECK:   %[[ZLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       CHECK:   %[[ZSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       CHECK:   scf.for %[[IV:.+]] = %[[ZLB]] to %[[UB]] step %[[ZSTEP]] {
//       CHECK:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>

// NO-BLOCK-DIM-LABEL: func.func @distribute_to_z
//  NO-BLOCK-DIM-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//   NO-BLOCK-DIM-DAG:   %[[ID:.+]] = gpu.thread_id z
//   NO-BLOCK-DIM-DAG:   %[[DIM:.+]] = arith.constant 64 : index
//       NO-BLOCK-DIM:   %[[ZLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       NO-BLOCK-DIM:   %[[ZSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       NO-BLOCK-DIM:   scf.for %[[IV:.+]] = %[[ZLB]] to %[[UB]] step %[[ZSTEP]] {
//       NO-BLOCK-DIM:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>


// -----

func.func @no_distribute_without_attr(%lb : index, %ub : index, %step: index, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32

  %init = tensor.empty() : tensor<2x128xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.store %zero, %output[%iv] : memref<?xf32>
  }

  return
}

// CHECK-LABEL: func.func @no_distribute_without_attr
//  CHECK-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//       CHECK:   scf.for %{{.+}} = %[[LB]] to %[[UB]] step %[[STEP]] {
