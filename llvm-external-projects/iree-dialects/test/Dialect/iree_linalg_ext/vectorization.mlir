// RUN: iree-dialects-opt --iree-linalg-ext-pack-op-vectorization --split-input-file %s | FileCheck %s

func.func @simple_KCRS_to_KCRSsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = iree_linalg_ext.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : (tensor<1x1x32x8xf32> tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @simple_KCRS_to_KCRSsr
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true, true, true]
// CHECK-SAME:    : tensor<1x1x32x8xf32>, vector<1x1x32x8xf32>
// CHECK:         %[[BCAST:.+]] = vector.broadcast %[[READ]] : vector<1x1x32x8xf32> to vector<1x1x1x1x32x8xf32>
// CHECK:         %[[TRANS:.+]] =  vector.transpose %[[BCAST]], [2, 3, 0, 1, 5, 4] : vector<1x1x1x1x32x8xf32> to vector<1x1x1x1x8x32xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[TRANS]],
// CHECK-SAME:      %[[EMPTY]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x8x32xf32>, tensor<1x1x1x1x8x32xf32>

// -----

func.func @simple_pad_and_pack(%input: tensor<5x1xf32>, %output: tensor<1x1x8x2xf32>, %pad: f32) -> tensor<1x1x8x2xf32> {
  %0 = iree_linalg_ext.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<5x1xf32> tensor<1x1x8x2xf32>) -> tensor<1x1x8x2xf32>
  return %0 : tensor<1x1x8x2xf32>
}
// CHECK-LABEL: func.func @simple_pad_and_pack
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[PAD:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK-SAME:      : tensor<5x1xf32>, vector<8x2xf32>
// CHECK:         %[[BCAST:.+]] = vector.broadcast %[[READ]] : vector<8x2xf32> to vector<1x1x8x2xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[BCAST]],
// CHECK-SAME:      %[[EMPTY]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:     {in_bounds = [true, true, true, true]} : vector<1x1x8x2xf32>, tensor<1x1x8x2xf32>

// -----

func.func @simple_NC_to_CNnc(%arg0: tensor<32x8xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>{
  %0 = iree_linalg_ext.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : (tensor<32x8xf32> tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_NC_to_CNnc
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]}
// CHECK-SAME:    : tensor<32x8xf32>, vector<32x8xf32>
// CHECK:         %[[BCAST:.+]] = vector.broadcast %[[READ]] : vector<32x8xf32> to vector<1x1x32x8xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[BCAST]],
// CHECK-SAME:      %[[EMPTY]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true, true, true]} : vector<1x1x32x8xf32>, tensor<1x1x32x8xf32>

// -----

// TODO(hanchung): Vectorize below cases
func.func @KCRS_to_KCRSsr(%arg0: tensor<1x1x128x64xf32>, %arg1: tensor<1x1x4x8x8x32xf32>) -> tensor<1x1x4x8x8x32xf32> {
  %0 = iree_linalg_ext.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : (tensor<1x1x128x64xf32> tensor<1x1x4x8x8x32xf32>) -> tensor<1x1x4x8x8x32xf32>
  return %0 : tensor<1x1x4x8x8x32xf32>
}
// CHECK: iree_linalg_ext.pack

// -----

func.func @pad_and_pack(%arg0: tensor<13x15xf32>, %arg1: tensor<2x8x8x2xf32>, %arg2: f32) -> tensor<2x8x8x2xf32> {
  %0 = iree_linalg_ext.pack %arg0 padding_value(%arg2 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : (tensor<13x15xf32> tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}
// CHECK: iree_linalg_ext.pack

// -----

func.func @NC_to_CNcn(%arg0: tensor<128x256xf32>, %arg1: tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32> {
  %0 = iree_linalg_ext.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : (tensor<128x256xf32> tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32>
  return %0 : tensor<32x4x32x8xf32>
}
// CHECK: iree_linalg_ext.pack
