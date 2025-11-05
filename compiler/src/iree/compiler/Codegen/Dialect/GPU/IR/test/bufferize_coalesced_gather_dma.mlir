// RUN: iree-opt %s --one-shot-bufferize="bufferize-function-boundaries" --split-input-file | FileCheck %s



// Test bufferization without indices (contiguous gather)
func.func @bufferize_coalesced_gather_dma_no_indices(%source: tensor<4x32xf32>,
                                                      %dest: tensor<4x32xf32>,
                                                      %lane: index) -> tensor<4x32xf32> {
  %result = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : tensor<4x32xf32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %result : tensor<4x32xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_no_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}} into %{{.+}} lane(%{{.+}}) : memref<4x32xf32{{.+}}>, memref<4x32xf32{{.+}}>, index

// -----

// Test bufferization with vector indices
func.func @bufferize_coalesced_gather_dma_with_indices(%idx0: vector<4xindex>,
                                                         %source: tensor<4x64xf32>,
                                                         %dest: tensor<4x64xf32>,
                                                         %lane: index) -> tensor<4x64xf32> {
  %result = iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : tensor<4x64xf32>, vector<4xindex>, tensor<4x64xf32>, index -> tensor<4x64xf32>
  return %result : tensor<4x64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_with_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : memref<4x64xf32{{.+}}>, vector<4xindex>, memref<4x64xf32{{.+}}>, index
