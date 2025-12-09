// RUN: iree-opt %s --one-shot-bufferize="bufferize-function-boundaries" --split-input-file | FileCheck %s



// Test bufferization without indices (contiguous gather).
func.func @bufferize_coalesced_gather_dma_no_indices(%source: tensor<4x32xf32>,
                                                      %dest: tensor<4x32xf32>,
                                                      %lane: index) -> tensor<4x32xf32> {
  %result = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : (tensor<4x32xf32>, tensor<4x32xf32>, index) -> tensor<4x32xf32>
  return %result : tensor<4x32xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_no_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}} into %{{.+}} lane(%{{.+}}) : (memref<4x32xf32{{.+}}>, memref<4x32xf32{{.+}}>, index) -> ()

// -----

// Test bufferization with vector indices.
func.func @bufferize_coalesced_gather_dma_with_indices(%idx0: vector<4xi32>,
                                                         %source: tensor<4x64xf32>,
                                                         %dest: tensor<4x64xf32>,
                                                         %lane: index) -> tensor<4x64xf32> {
  %result = iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : (tensor<4x64xf32>, vector<4xi32>, tensor<4x64xf32>, index) -> tensor<4x64xf32>
  return %result : tensor<4x64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_with_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : (memref<4x64xf32{{.+}}>, vector<4xi32>, memref<4x64xf32{{.+}}>, index) -> ()

// -----

// Test bufferization with tensor indices.
func.func @bufferize_coalesced_gather_dma_tensor_indices(%idx0: tensor<4xi32>,
                                                          %source: tensor<4x64xf32>,
                                                          %dest: tensor<4x64xf32>,
                                                          %lane: index) -> tensor<4x64xf32> {
  %result = iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : (tensor<4x64xf32>, tensor<4xi32>, tensor<4x64xf32>, index) -> tensor<4x64xf32>
  return %result : tensor<4x64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_tensor_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : (memref<4x64xf32{{.+}}>, memref<4xi32{{.+}}>, memref<4x64xf32{{.+}}>, index) -> ()

// -----

// Test bufferization with 1D tensors
func.func @bufferize_coalesced_gather_dma_1d(%source: tensor<1024xf32>,
                                              %dest: tensor<1024xf32>,
                                              %lane: index) -> tensor<1024xf32> {
  %result = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : (tensor<1024xf32>, tensor<1024xf32>, index) -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_1d
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}} into %{{.+}} lane(%{{.+}}) : (memref<1024xf32{{.+}}>, memref<1024xf32{{.+}}>, index) -> ()

// -----

// Test bufferization with multiple tensor indices.
func.func @bufferize_coalesced_gather_dma_multiple_indices(%idx0: tensor<4xi32>,
                                                             %idx1: tensor<4xi32>,
                                                             %source: tensor<64x128xf32>,
                                                             %dest: tensor<4x128xf32>,
                                                             %lane: index) -> tensor<4x128xf32> {
  %result = iree_gpu.coalesced_gather_dma %source[%idx0, %idx1] into %dest lane(%lane)
    : (tensor<64x128xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4x128xf32>, index) -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_multiple_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}, %{{.+}}] into %{{.+}} lane(%{{.+}}) : (memref<64x128xf32{{.+}}>, memref<4xi32{{.+}}>, memref<4xi32{{.+}}>, memref<4x128xf32{{.+}}>, index) -> ()

// -----

// Test bufferization with slice semantics (offsets/sizes/strides).
// The slice should become a memref.subview of the destination.
func.func @bufferize_coalesced_gather_dma_with_slice(%source: tensor<1x1x64xf32>,
                                                      %dest: tensor<2x2x64xf32>,
                                                      %off0: index, %off1: index,
                                                      %lane: index) -> tensor<2x2x64xf32> {
  %result = iree_gpu.coalesced_gather_dma %source into %dest [%off0, %off1, 0] [1, 1, 64] [1, 1, 1] lane(%lane)
    : (tensor<1x1x64xf32>, tensor<2x2x64xf32>, index, index, index) -> tensor<2x2x64xf32>
  return %result : tensor<2x2x64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_with_slice
//  CHECK-SAME:   %[[SOURCE:.+]]: memref<1x1x64xf32{{.+}}>, %[[DEST:.+]]: memref<2x2x64xf32{{.+}}>, %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[LANE:.+]]: index
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[DEST]][%[[OFF0]], %[[OFF1]], 0] [1, 1, 64] [1, 1, 1]
//       CHECK:   iree_gpu.coalesced_gather_dma %[[SOURCE]] into %[[SUBVIEW]] lane(%[[LANE]]) : (memref<1x1x64xf32{{.+}}>, memref<1x1x64xf32{{.+}}>, index) -> ()
