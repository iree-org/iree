// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-promote-matmul-operands))" | FileCheck %s

func.func @matmul(%a: tensor<32x1024xf32>, %b: tensor<1024x128xf32>) -> tensor<32x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%fill : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %mm : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @matmul
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<32x1024xf32>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<1024x128xf32>
//   CHECK-DAG:   %[[PA:.+]] = linalg.copy {{.*}} ins(%[[A]] : tensor<32x1024xf32>)
//   CHECK-DAG:   %[[PB:.+]] = linalg.copy {{.*}} ins(%[[B]] : tensor<1024x128xf32>)
//       CHECK:   linalg.matmul ins(%[[PA]], %[[PB]] : tensor<32x1024xf32>, tensor<1024x128xf32>)

// -----

func.func @matvec(%a: tensor<1x1024xf32>, %b: tensor<1024x128xf32>) -> tensor<1x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x128xf32>) -> tensor<1x128xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<1x1024xf32>, tensor<1024x128xf32>) outs(%fill : tensor<1x128xf32>) -> tensor<1x128xf32>
  return %mm : tensor<1x128xf32>
}

// Verify that no copies are generated for matvec operations.
// CHECK-LABEL: func.func @matvec
//   CHECK-NOT:   linalg.copy
//       CHECK: return
