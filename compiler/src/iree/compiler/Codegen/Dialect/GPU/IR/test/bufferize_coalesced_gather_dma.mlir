// RUN: iree-opt %s --one-shot-bufferize="bufferize-function-boundaries" --split-input-file | FileCheck %s



// Test bufferization without indices (contiguous gather).
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

// Test bufferization with vector indices.
func.func @bufferize_coalesced_gather_dma_with_indices(%idx0: vector<4xi32>,
                                                         %source: tensor<4x64xf32>,
                                                         %dest: tensor<4x64xf32>,
                                                         %lane: index) -> tensor<4x64xf32> {
  %result = iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : tensor<4x64xf32>, vector<4xi32>, tensor<4x64xf32>, index -> tensor<4x64xf32>
  return %result : tensor<4x64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_with_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : memref<4x64xf32{{.+}}>, vector<4xi32>, memref<4x64xf32{{.+}}>, index

// -----

// Test bufferization with tensor indices.
func.func @bufferize_coalesced_gather_dma_tensor_indices(%idx0: tensor<4xi32>,
                                                          %source: tensor<4x64xf32>,
                                                          %dest: tensor<4x64xf32>,
                                                          %lane: index) -> tensor<4x64xf32> {
  %result = iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : tensor<4x64xf32>, tensor<4xi32>, tensor<4x64xf32>, index -> tensor<4x64xf32>
  return %result : tensor<4x64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_tensor_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : memref<4x64xf32{{.+}}>, memref<4xi32{{.+}}>, memref<4x64xf32{{.+}}>, index

// -----

// Test bufferization with 1D tensors
func.func @bufferize_coalesced_gather_dma_1d(%source: tensor<1024xf32>,
                                              %dest: tensor<1024xf32>,
                                              %lane: index) -> tensor<1024xf32> {
  %result = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : tensor<1024xf32>, tensor<1024xf32>, index -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_1d
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}} into %{{.+}} lane(%{{.+}}) : memref<1024xf32{{.+}}>, memref<1024xf32{{.+}}>, index

// -----

// Test bufferization with multiple tensor indices.
func.func @bufferize_coalesced_gather_dma_multiple_indices(%idx0: tensor<4xi32>,
                                                             %idx1: tensor<4xi32>,
                                                             %source: tensor<64x128xf32>,
                                                             %dest: tensor<4x128xf32>,
                                                             %lane: index) -> tensor<4x128xf32> {
  %result = iree_gpu.coalesced_gather_dma %source[%idx0, %idx1] into %dest lane(%lane)
    : tensor<64x128xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4x128xf32>, index -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_multiple_indices
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}, %{{.+}}] into %{{.+}} lane(%{{.+}}) : memref<64x128xf32{{.+}}>, memref<4xi32{{.+}}>, memref<4xi32{{.+}}>, memref<4x128xf32{{.+}}>, index

// -----

// Test bufferization with in_bounds attribute (for fused tensor.pad).
func.func @bufferize_coalesced_gather_dma_in_bounds(%source: tensor<4x32xf32>,
                                                     %dest: tensor<4x64xf32>,
                                                     %lane: index) -> tensor<4x64xf32> {
  %result = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    in_bounds [true, false]
    : tensor<4x32xf32>, tensor<4x64xf32>, index -> tensor<4x64xf32>
  return %result : tensor<4x64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_in_bounds
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}} into %{{.+}} lane(%{{.+}}) in_bounds [true, false] : memref<4x32xf32{{.+}}>, memref<4x64xf32{{.+}}>, index

// -----

// Test bufferization with slice semantics (offsets/sizes/strides) inside forall.
func.func @bufferize_coalesced_gather_dma_slice(%source: tensor<2x2x64xf32>) -> tensor<2x2x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %empty = tensor.empty() : tensor<2x2x64xf32>
  %result = scf.forall (%tid0, %tid1, %tid2) in (2, 2, 64) shared_outs(%init = %empty) -> tensor<2x2x64xf32> {
    %src_slice = tensor.extract_slice %source[%tid0, %tid1, 0] [1, 1, 64] [1, 1, 1]
      : tensor<2x2x64xf32> to tensor<1x1x64xf32>
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %src_slice into %init
        [%tid0, %tid1, %c0] [%c1, %c1, %c64] [%c1, %c1, %c1]
        lane(%tid2)
        : tensor<1x1x64xf32>, tensor<2x2x64xf32>, index, index, index, index, index, index, index, index, index, index
    }
  } {mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return %result : tensor<2x2x64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_slice
//   CHECK-DAG:   %[[SRC_SUBVIEW:.+]] = memref.subview %{{.+}}[%{{.+}}, %{{.+}}, 0] [1, 1, 64] [1, 1, 1] : memref<2x2x64xf32{{.+}}> to memref<1x1x64xf32
//   CHECK-DAG:   %[[DST_SUBVIEW:.+]] = memref.subview %{{.+}}[%{{.+}}, %{{.+}}, 0] [1, 1, 64] [1, 1, 1] : memref<2x2x64xf32> to memref<1x1x64xf32
//       CHECK:   iree_gpu.coalesced_gather_dma %[[SRC_SUBVIEW]] into %[[DST_SUBVIEW]] lane(%{{.+}}) : memref<1x1x64xf32{{.+}}>, memref<1x1x64xf32{{.+}}>, index
