// RUN: iree-opt -split-input-file -iree-spirv-distribute -mlir-print-local-scope %s | FileCheck %s

func @distribute_to_x(%lb : index, %ub : index, %step: index, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32

  %init = linalg.init_tensor [2, 128] : tensor<2x128xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.store %zero, %output[%iv] : memref<?xf32>
  } {iree.spirv.distribute_dim = 0 : index}

  return
}

// CHECK-LABEL: func @distribute_to_x
//  CHECK-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//       CHECK:   %[[ID:.+]] = "gpu.thread_id"() {dimension = "x"}
//       CHECK:   %[[DIM:.+]] = "gpu.block_dim"() {dimension = "x"}
//       CHECK:   %[[XLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       CHECK:   %[[XSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       CHECK:   scf.for %[[IV:.+]] = %[[XLB]] to %[[UB]] step %[[XSTEP]] {
//       CHECK:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>

// -----

func @distribute_to_y(%lb : index, %ub : index, %step: index, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32

  %init = linalg.init_tensor [2, 128] : tensor<2x128xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.store %zero, %output[%iv] : memref<?xf32>
  } {iree.spirv.distribute_dim = 1 : index}

  return
}

// CHECK-LABEL: func @distribute_to_y
//  CHECK-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//       CHECK:   %[[ID:.+]] = "gpu.thread_id"() {dimension = "y"}
//       CHECK:   %[[DIM:.+]] = "gpu.block_dim"() {dimension = "y"}
//       CHECK:   %[[YLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       CHECK:   %[[YSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       CHECK:   scf.for %[[IV:.+]] = %[[YLB]] to %[[UB]] step %[[YSTEP]] {
//       CHECK:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>

// -----

func @distribute_to_z(%lb : index, %ub : index, %step: index, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32

  %init = linalg.init_tensor [2, 128] : tensor<2x128xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.store %zero, %output[%iv] : memref<?xf32>
  } {iree.spirv.distribute_dim = 2 : index}

  return
}

// CHECK-LABEL: func @distribute_to_z
//  CHECK-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//       CHECK:   %[[ID:.+]] = "gpu.thread_id"() {dimension = "z"}
//       CHECK:   %[[DIM:.+]] = "gpu.block_dim"() {dimension = "z"}
//       CHECK:   %[[ZLB:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[ID]], %[[STEP]], %[[LB]]]
//       CHECK:   %[[ZSTEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[DIM]], %[[STEP]]]
//       CHECK:   scf.for %[[IV:.+]] = %[[ZLB]] to %[[UB]] step %[[ZSTEP]] {
//       CHECK:     memref.store %{{.+}}, %{{.+}}[%[[IV]]] : memref<?xf32>

// -----

func @no_distribute_without_attr(%lb : index, %ub : index, %step: index, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32

  %init = linalg.init_tensor [2, 128] : tensor<2x128xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.store %zero, %output[%iv] : memref<?xf32>
  }

  return
}

// CHECK-LABEL: func @no_distribute_without_attr
//  CHECK-SAME: %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index
//       CHECK:   scf.for %{{.+}} = %[[LB]] to %[[UB]] step %[[STEP]] {
