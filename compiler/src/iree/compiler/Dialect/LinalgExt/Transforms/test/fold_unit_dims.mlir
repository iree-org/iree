// RUN: iree-opt --pass-pipeline="builtin.module(iree-linalg-ext-fold-unit-extent-dims)" %s --split-input-file --mlir-print-local-scope | FileCheck %s

util.func public @gather_unit_batch_dims(%source: tensor<4x4x4x4xf16>, %indices: tensor<4x1x2xi64>) -> tensor<4x1x4x4xf16> {
  %empty = tensor.empty() : tensor<4x1x4x4xf16>
  %0 = iree_linalg_ext.gather dimension_map = [0, 1]
          ins(%source, %indices: tensor<4x4x4x4xf16>, tensor<4x1x2xi64>)
          outs(%empty: tensor<4x1x4x4xf16>) -> tensor<4x1x4x4xf16>
  util.return %0 : tensor<4x1x4x4xf16>
}

// CHECK-LABEL: util.func public @gather_unit_batch_dims
// CHECK: %[[INDICES:.+]] = tensor.extract_slice %{{.*}}[0, 0, 0] [4, 1, 2] [1, 1, 1] : tensor<4x1x2xi64> to tensor<4x2xi64>
// CHECK: %[[OUTPUT:.+]] = tensor.extract_slice %{{.*}}[0, 0, 0, 0] [4, 1, 4, 4] [1, 1, 1, 1] : tensor<4x1x4x4xf16> to tensor<4x4x4xf16>
// CHECK: %[[RESULT:.+]] = iree_linalg_ext.gather
// CHECK-SAME: -> tensor<4x4x4xf16>
// CHECK: %[[EXPANDED_RESULT:.+]] = tensor.insert_slice %[[RESULT]] into %{{.*}}[0, 0, 0, 0] [4, 1, 4, 4] [1, 1, 1, 1] : tensor<4x4x4xf16> into tensor<4x1x4x4xf16>

// -----

util.func public @gather_batch_and_slice_dims(%source: tensor<4x4x1x4xf16>, %indices: tensor<4x1x2xi64>) -> tensor<4x1x1x4xf16> {
  %empty = tensor.empty() : tensor<4x1x1x4xf16>
  %0 = iree_linalg_ext.gather dimension_map = [0, 1]
          ins(%source, %indices: tensor<4x4x1x4xf16>, tensor<4x1x2xi64>)
          outs(%empty: tensor<4x1x1x4xf16>) -> tensor<4x1x1x4xf16>
  util.return %0 : tensor<4x1x1x4xf16>
}

// CHECK-LABEL: util.func public @gather_batch_and_slice_dims
// CHECK-SAME: %[[SOURCE:.+]]: tensor<4x4x1x4xf16>, %[[INDICES:.+]]: tensor<4x1x2xi64>
// CHECK: %[[OUTPUT:.+]] = tensor.empty() : tensor<4x1x1x4xf16>
// CHECK: %[[INDICES_BATCH:.+]] = tensor.extract_slice %[[INDICES]]
// CHECK-SAME: tensor<4x1x2xi64> to tensor<4x2xi64>
// CHECK: %[[OUTPUT_BATCH:.+]] = tensor.extract_slice %[[OUTPUT]]
// CHECK-SAME: tensor<4x1x1x4xf16> to tensor<4x1x4xf16>
// CHECK: %[[SOURCE_SLICE:.+]] = tensor.extract_slice %[[SOURCE]]
// CHECK-SAME: tensor<4x4x1x4xf16> to tensor<4x4x4xf16>
// CHECK: %[[OUTPUT_SLICE:.+]] = tensor.extract_slice %[[OUTPUT_BATCH]]
// CHECK-SAME: tensor<4x1x4xf16> to tensor<4x4xf16>
// CHECK: iree_linalg_ext.gather
// CHECK-SAME: ins(%[[SOURCE_SLICE]], %[[INDICES_BATCH]]
// CHECK-SAME: outs(%[[OUTPUT_SLICE]]
