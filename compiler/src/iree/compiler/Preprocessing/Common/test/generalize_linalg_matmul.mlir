// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))" --verify-each --split-input-file %s | FileCheck %s

util.func public @generalize_matmul(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
  %0 = tensor.empty() : tensor<1x128x128xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1: tensor<1x128x128xf32>, tensor<1x128x128xf32>) outs(%0 : tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  util.return %1 : tensor<1x128x128xf32>
}

// CHECK-LABEL: util.func public @generalize_matmul
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<1x128x128xf32>, %[[ARG1:.+]]: tensor<1x128x128xf32>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:   %[[ARG0]], %[[ARG1]]
