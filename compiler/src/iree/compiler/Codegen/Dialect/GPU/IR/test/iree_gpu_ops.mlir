// RUN: iree-opt %s --split-input-file | FileCheck %s

func.func @shuffle_tensor(%init: memref<6x6xf32>, %arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %arg0[0, 0] [2, 3] [1, 1] to %init[0, 0] [3, 2] [1, 1] : tensor<2x3xf32> -> memref<6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor %arg1[0, 0] [2, 3] [1, 1] to
//  CHECK-SAME:     %arg0 [0, 0] [3, 2] [1, 1] : tensor<2x3xf32> -> memref<6x6xf32> -> tensor<3x2xf32>

// -----

func.func @rank_reducing_shuffle_tensor(%init: memref<1x6x6xf32>, %arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %arg0[0, 0, 0] [1, 2, 3] [1, 1, 1] to %init[0, 0, 0] [1, 3, 2] [1, 1, 1] : tensor<2x3xf32> -> memref<1x6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @rank_reducing_shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor %arg1[0, 0, 0] [1, 2, 3] [1, 1, 1] to
//  CHECK-SAME:     %arg0 [0, 0, 0] [1, 3, 2] [1, 1, 1] : tensor<2x3xf32> -> memref<1x6x6xf32> -> tensor<3x2xf32>

// -----

func.func @dynamic_alloc_shuffle_tensor(%init: memref<?x?xf32>, %arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %arg0[0, 0] [2, 3] [1, 1] to %init[0, 0] [3, 2] [1, 1] : tensor<2x3xf32> -> memref<?x?xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @dynamic_alloc_shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor %arg1[0, 0] [2, 3] [1, 1] to
//  CHECK-SAME:     %arg0 [0, 0] [3, 2] [1, 1] : tensor<2x3xf32> -> memref<?x?xf32> -> tensor<3x2xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @vector_multi_mma(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_multi_mma(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @single_multi_mma(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : vector<4xf16>, vector<4xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}
