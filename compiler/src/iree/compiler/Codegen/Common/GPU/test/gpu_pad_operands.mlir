// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-pad-operands))" | FileCheck %s

#lowering_config = #iree_gpu.lowering_config<{padding = [3, 7, 11]}>

func.func @matmul(%a: tensor<32x1024xf32>, %b: tensor<1024x128xf32>) -> tensor<32x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%fill : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %mm : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @matmul
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<32x1024xf32>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<1024x128xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill {{.*}} -> tensor<32x128xf32>
//       CHECK:   %[[PADDED_LHS:.+]] = tensor.pad %[[A]] low[0, 0] high[1, 10]
//       CHECK:   %[[PADDED_RHS:.+]] = tensor.pad %[[B]] low[0, 0] high[10, 5]
//       CHECK:   %[[PADDED_INIT:.+]] = tensor.pad %[[FILL]] low[0, 0] high[1, 5]
//       CHECK:   %[[PADDED_RESULT:.+]] = linalg.matmul
//  CHECK-SAME:     ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<33x1034xf32>, tensor<1034x133xf32>)
//  CHECK-SAME:     outs(%[[PADDED_INIT]] : tensor<33x133xf32>) -> tensor<33x133xf32>
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[PADDED_RESULT]][0, 0] [32, 128] [1, 1]
//  CHECK-SAME:     : tensor<33x133xf32> to tensor<32x128xf32>
//       CHECK:   return %[[EXTRACT]] : tensor<32x128xf32>
