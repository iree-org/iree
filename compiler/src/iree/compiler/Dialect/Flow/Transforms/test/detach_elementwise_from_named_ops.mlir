// RUN: iree-opt --split-input-file --iree-flow-detach-elementwise-from-named-ops --mlir-print-local-scope %s | FileCheck %s

func.func @matmul(%a: tensor<?x64xf32>, %b: tensor<64x?xf32>, %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%a, %b : tensor<?x64xf32>, tensor<64x?xf32>) outs(%c : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul
//  CHECK-SAME: (%[[A:.+]]: tensor<?x64xf32>, %[[B:.+]]: tensor<64x?xf32>, %[[C:.+]]: tensor<?x?xf32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

//       CHECK:   %[[DIM0:.+]] = tensor.dim %[[C]], %[[C0]]
//       CHECK:   %[[DIM1:.+]] = tensor.dim %[[C]], %[[C1]]
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[DIM0]], %[[DIM1]]]
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[F0]] : f32) outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:   %[[MM:.+]] = linalg.matmul
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<?x64xf32>, tensor<64x?xf32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<?x?xf32>)
//       CHECK:   %[[EW:.+]] = linalg.generic {
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]}
//  CHECK-SAME:     ins(%[[MM]], %[[C]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<?x?xf32>)
//       CHECK:   ^{{.+}}(%[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32, %{{.+}}: f32):
//       CHECK:     %[[ADD:.+]] = arith.addf %[[ARG0]], %[[ARG1]] : f32
//       CHECK:     linalg.yield %[[ADD]] : f32
//       CHECK:   return %[[EW]]

// -----

func.func @batch_matmul(%a: tensor<?x8x?xi32>, %b: tensor<?x?x16xi32>, %c: tensor<?x8x16xi32>) -> tensor<?x8x16xi32> {
  %0 = linalg.batch_matmul ins(%a, %b : tensor<?x8x?xi32>, tensor<?x?x16xi32>) outs(%c : tensor<?x8x16xi32>) -> tensor<?x8x16xi32>
  return %0 : tensor<?x8x16xi32>
}

// CHECK-LABEL: func @batch_matmul
//  CHECK-SAME: (%[[A:.+]]: tensor<?x8x?xi32>, %[[B:.+]]: tensor<?x?x16xi32>, %[[C:.+]]: tensor<?x8x16xi32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[I0:.+]] = arith.constant 0 : i32

//       CHECK:   %[[DIM0:.+]] = tensor.dim %[[C]], %[[C0]] : tensor<?x8x16xi32>
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[DIM0]], 8, 16] : tensor<?x8x16xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[I0]] : i32) outs(%[[INIT]] : tensor<?x8x16xi32>) -> tensor<?x8x16xi32>
//       CHECK:   %[[MM:.+]] = linalg.batch_matmul
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<?x8x?xi32>, tensor<?x?x16xi32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<?x8x16xi32>)
//       CHECK:   %[[EW:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[MM]], %arg2 : tensor<?x8x16xi32>, tensor<?x8x16xi32>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<?x8x16xi32>)
//       CHECK:     %[[ADD:.+]] = arith.addi
//       CHECK:     linalg.yield %[[ADD]] : i32
//       CHECK:   return %[[EW]]
