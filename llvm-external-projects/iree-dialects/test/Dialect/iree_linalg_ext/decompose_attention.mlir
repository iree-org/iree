// RUN: iree-dialects-opt --split-input-file -iree-linalg-ext-decompose-attention -cse %s | FileCheck %s

func.func @attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}

// CHECK:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = tensor.empty() : tensor<192x64x1024xf32>
// CHECK:        %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<192x1024x64xf32>) outs(%[[D1]] :
// CHECK-SAME:     tensor<192x64x1024xf32>) permutation = [0, 2, 1]
// CHECK:        %[[D2:.+]] = tensor.empty() : tensor<192x1024x1024xf32>
// CHECK:        %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D3:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D2]] : tensor<192x1024x1024xf32>) ->
// CHECK-SAME:     tensor<192x1024x1024xf32>
// CHECK:        %[[D4:.+]] = linalg.batch_matmul ins(%[[ARG0]], %[[TRANSPOSED]] : tensor<192x1024x64xf32>,
// CHECK-SAME:     tensor<192x64x1024xf32>) outs(%[[D3]] : tensor<192x1024x1024xf32>) -> tensor<192x1024x1024xf32>
// CHECK:        %[[D5:.+]] = iree_linalg_ext.softmax dimension(2) ins(%[[D4]] : tensor<192x1024x1024xf32>) outs(%[[D2]]
// CHECK-SAME:     : tensor<192x1024x1024xf32>) -> tensor<192x1024x1024xf32>
// CHECK:        %[[D6:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D0]] : tensor<192x1024x64xf32>) ->
// CHECK-SAME:     tensor<192x1024x64xf32>
// CHECK:        %[[D7:.+]] = linalg.batch_matmul ins(%[[D5]], %[[ARG2]] : tensor<192x1024x1024xf32>,
// CHECK-SAME:     tensor<192x1024x64xf32>) outs(%[[D6]] : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK:        return %[[D7]] : tensor<192x1024x64xf32>
// CHECK:      }
