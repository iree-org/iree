// RUN: iree-dialects-opt --iree-linalg-ext-vectorization --split-input-file %s | FileCheck %s

func.func @KCRS_to_KCRSsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = iree_linalg_ext.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : (tensor<1x1x32x8xf32> tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @KCRS_to_KCRSsr
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true, true, true]}
// CHECK-SAME:    : tensor<1x1x32x8xf32>, vector<1x1x32x8xf32>
// CHECK:         %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<1x1x32x8xf32> to vector<1x1x1x32x1x8xf32>
// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty
// CHECK:         %[[TRANS:.+]] = vector.transpose %[[CAST]], [0, 1, 2, 4, 5, 3]
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[TRANS:[A-Za-z0-9]+]],
// CHECK-SAME:      %[[EMPTY]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x8x32xf32>, tensor<1x1x1x1x8x32xf32>

// -----

func.func @pad_and_pack(%input: tensor<5x1xf32>, %output: tensor<1x1x8x2xf32>, %pad: f32) -> tensor<1x1x8x2xf32> {
  %0 = iree_linalg_ext.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<5x1xf32> tensor<1x1x8x2xf32>) -> tensor<1x1x8x2xf32>
  return %0 : tensor<1x1x8x2xf32>
}

// CHECK-LABEL: func.func @pad_and_pack
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[PAD:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK-SAME:    : tensor<5x1xf32>, vector<8x2xf32>
// CHECK:         %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<8x2xf32> to vector<1x8x1x2xf32>
// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty
// CHECK:         %[[TRANS:.+]] = vector.transpose %[[CAST]], [0, 2, 1, 3]
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[TRANS:[A-Za-z0-9]+]],
// CHECK-SAME:      %[[EMPTY]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:     {in_bounds = [true, true, true, true]} : vector<1x1x8x2xf32>, tensor<1x1x8x2xf32>

// -----

func.func @NC_to_CNnc(%arg0: tensor<32x8xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>{
  %0 = iree_linalg_ext.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : (tensor<32x8xf32> tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}

// CHECK-LABEL: func.func @NC_to_CNnc
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]}
// CHECK-SAME:    : tensor<32x8xf32>, vector<32x8xf32>
// CHECK:         %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<32x8xf32> to vector<1x32x1x8xf32>
// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty
// CHECK:         %[[TRANS:.+]] = vector.transpose %[[CAST]], [2, 0, 1, 3]
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[TRANS:[A-Za-z0-9]+]],
// CHECK-SAME:      %[[EMPTY]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true, true, true]} : vector<1x1x32x8xf32>, tensor<1x1x32x8xf32>
