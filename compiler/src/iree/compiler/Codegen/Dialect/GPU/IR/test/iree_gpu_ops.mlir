// RUN: iree-opt %s --split-input-file | FileCheck %s

func.func @shuffle_tensor(%init: tensor<6x6xf32>, %source: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[0, 0] [2, 3] [1, 1] to %init {
  ^bb0(%intermediate: tensor<6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, 0] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<2x3xf32> -> tensor<6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor %arg1[0, 0] [2, 3] [1, 1] to %arg0 {
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<6x6xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][0, 0] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
//       CHECK:       iree_gpu.yield %[[SLICE]] : tensor<3x2xf32>
//       CHECK:   } : tensor<2x3xf32> -> tensor<6x6xf32> -> tensor<3x2xf32>

// -----

func.func @rank_reducing_shuffle_tensor(%init: tensor<1x6x6xf32>, %source: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[0, 0, 0] [1, 2, 3] [1, 1, 1] to %init {
  ^bb0(%intermediate: tensor<1x6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, 0, 0] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<2x3xf32> -> tensor<1x6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @rank_reducing_shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor %arg1[0, 0, 0] [1, 2, 3] [1, 1, 1] to %arg0 {
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<1x6x6xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][0, 0, 0] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
//       CHECK:       iree_gpu.yield %[[SLICE]] : tensor<3x2xf32>
//       CHECK:   } : tensor<2x3xf32> -> tensor<1x6x6xf32> -> tensor<3x2xf32>

// -----

func.func @reshape_shuffle_tensor(%init: tensor<12x12xf32>, %source: tensor<2x3xf32>) -> tensor<2x1x3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[0, 0] [2, 3] [1, 1] to %init {
  ^bb0(%intermediate: tensor<12x12xf32>):
    %expand = tensor.expand_shape %intermediate [[0, 1], [2, 3]] output_shape [4, 3, 3, 4] : tensor<12x12xf32> into tensor<4x3x3x4xf32>
    %slice = tensor.extract_slice %expand[0, 0, 0, 0] [2, 1, 3, 2] [1, 1, 1, 1] : tensor<4x3x3x4xf32> to tensor<2x1x3x2xf32>
    iree_gpu.yield %slice : tensor<2x1x3x2xf32>
  } : tensor<2x3xf32> -> tensor<12x12xf32> -> tensor<2x1x3x2xf32>
  return %0 : tensor<2x1x3x2xf32>
}

// CHECK-LABEL: func @reshape_shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor
//       CHECK:       tensor.expand_shape
//       CHECK:       tensor.extract_slice
//       CHECK:   } : tensor<2x3xf32> -> tensor<12x12xf32> -> tensor<2x1x3x2xf32>
