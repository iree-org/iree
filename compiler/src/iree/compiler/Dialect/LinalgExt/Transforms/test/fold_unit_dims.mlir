// RUN: iree-opt --pass-pipeline="builtin.module(iree-linalg-ext-fold-unit-extent-dims)" %s --split-input-file --mlir-print-local-scope | FileCheck %s

util.func public @gather(%source: tensor<4x4x4x4xf16>, %indices: tensor<4x1x2xi64>) -> tensor<4x1x4x4xf16> {
  %empty = tensor.empty() : tensor<4x1x4x4xf16>
  %0 = iree_linalg_ext.gather dimension_map = [0, 1]
          ins(%source, %indices: tensor<4x4x4x4xf16>, tensor<4x1x2xi64>)
          outs(%empty: tensor<4x1x4x4xf16>) -> tensor<4x1x4x4xf16>
  util.return %0 : tensor<4x1x4x4xf16>
}

// CHECK-LABEL: util.func public @gather
// CHECK: %[[INDICES:.+]] = tensor.extract_slice %{{.*}}[0, 0, 0] [4, 1, 2] [1, 1, 1] : tensor<4x1x2xi64> to tensor<4x2xi64>
// CHECK: %[[OUTPUT:.+]] = tensor.extract_slice %{{.*}}[0, 0, 0, 0] [4, 1, 4, 4] [1, 1, 1, 1] : tensor<4x1x4x4xf16> to tensor<4x4x4xf16>
// CHECK: %[[RESULT:.+]] = iree_linalg_ext.gather
// CHECK-SAME: -> tensor<4x4x4xf16>
// CHECK: %[[EXPANDED_RESULT:.+]] = tensor.insert_slice %[[RESULT]] into %{{.*}}[0, 0, 0, 0] [4, 1, 4, 4] [1, 1, 1, 1] : tensor<4x4x4xf16> into tensor<4x1x4x4xf16>
