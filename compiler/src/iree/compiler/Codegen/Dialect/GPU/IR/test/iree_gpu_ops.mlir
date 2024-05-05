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
