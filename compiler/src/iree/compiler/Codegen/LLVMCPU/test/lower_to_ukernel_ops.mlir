// RUN: iree-opt --iree-llvmcpu-lower-to-ukernels %s | FileCheck %s

func.func @matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func @matmul(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "vmvx.matmul.f32f32f32"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[D0]], %[[D1]], %[[D2]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]] : tensor<?x?xf32>

// -----

func.func @matmul_fill(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %n = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func @matmul_fill(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 0 : i32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "vmvx.matmul.f32f32f32"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
// CHECK-SAME:       (%[[D0]], %[[D1]], %[[D2]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]] : tensor<?x?xf32>

// -----

func.func @matmul_i8i8i32(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?x?xi8>,
    %arg2 : tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?x?xi8>)
      outs(%arg2 : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
//      CHECK: func @matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xi8>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xi32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "vmvx.matmul.i8i8i32"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[D0]], %[[D1]], %[[D2]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @mmt4d(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>,
    %arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: func @mmt4d(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.mmt4d
// CHECK-SAME:       lhs(%[[ARG0]] :
// CHECK-SAME:       rhs(%[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       accumulate(true)
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @mmt4d_fill(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %m = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf32>
  %n = tensor.dim %arg1, %c1 : tensor<?x?x?x?xf32>
  %m0 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf32>
  %n0 = tensor.dim %arg1, %c2 : tensor<?x?x?x?xf32>
  %empty = tensor.empty(%m, %n, %m0, %n0) : tensor<?x?x?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%fill : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: func @mmt4d_fill(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//      CHECK:   %[[DST:.+]] = tensor.empty
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.mmt4d
// CHECK-SAME:       lhs(%[[ARG0]] :
// CHECK-SAME:       rhs(%[[ARG1]] :
// CHECK-SAME:       outs(%[[DST]] :
// CHECK-SAME:       accumulate(false)
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @mmt4d_i8i8i32(%arg0 : tensor<?x?x?x?xi8>, %arg1 : tensor<?x?x?x?xi8>,
    %arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>)
      outs(%arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}
//      CHECK: func @mmt4d_i8i8i32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi8>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi32>
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.mmt4d
// CHECK-SAME:       lhs(%[[ARG0]] :
// CHECK-SAME:       rhs(%[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       accumulate(true)
//      CHECK:   return %[[MICRO_KERNEL]]
