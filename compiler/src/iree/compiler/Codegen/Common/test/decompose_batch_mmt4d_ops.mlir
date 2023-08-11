// RUN: iree-opt --iree-codegen-decompose-batch-mmt4d-ops --split-input-file %s | FileCheck %s

func.func @batch_mmt4d_with_fill(%arg0: tensor<1x10x32x8x1xf32>, %arg1: tensor<1x80x32x4x1xf32>, %arg2: tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x10x32x8x1xf32>, tensor<1x80x32x4x1xf32>) outs(%0 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  return %1 : tensor<1x10x80x8x4xf32>
}

// CHECK:      func.func @batch_mmt4d_with_fill
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x10x32x8x1xf32>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x80x32x4x1xf32>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<1x10x80x8x4xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<1x10x80x8x4xf32> to tensor<10x80x8x4xf32>
// CHECK:        %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EXT_OUT]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<1x10x32x8x1xf32> to tensor<10x32x8x1xf32>
// CHECK-DAG:    %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<1x80x32x4x1xf32> to tensor<80x32x4x1xf32>
// CHECK:        %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXT_LHS]], %[[EXT_RHS]] : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%[[FILL]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xf32> into tensor<1x10x80x8x4xf32>
// CHECK:        return %[[INS]] : tensor<1x10x80x8x4xf32>

// -----

func.func @batch_mmt4d_with_no_fill(%arg0: tensor<1x10x32x8x1xf32>, %arg1: tensor<1x80x32x4x1xf32>, %arg2: tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32> {
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x10x32x8x1xf32>, tensor<1x80x32x4x1xf32>) outs(%arg2 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  return %1 : tensor<1x10x80x8x4xf32>
}

// CHECK:      func.func @batch_mmt4d_with_no_fill
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x10x32x8x1xf32>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x80x32x4x1xf32>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<1x10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<1x10x80x8x4xf32> to tensor<10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<1x10x32x8x1xf32> to tensor<10x32x8x1xf32>
// CHECK-DAG:    %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<1x80x32x4x1xf32> to tensor<80x32x4x1xf32>
// CHECK:        %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXT_LHS]], %[[EXT_RHS]] : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%[[EXT_OUT]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xf32> into tensor<1x10x80x8x4xf32>
// CHECK:        return %[[INS]] : tensor<1x10x80x8x4xf32>
