// RUN: iree-opt %s --one-shot-bufferize="bufferize-function-boundaries" --split-input-file | FileCheck %s

// Test bufferization of coalesced_gather_dma with static shapes
func.func @bufferize_coalesced_gather_dma_static(%idx0: vector<64xindex>,
                                                  %source: tensor<2048xf32>,
                                                  %dest: tensor<64xf32> {bufferization.writable = true},
                                                  %lane: index) -> tensor<64xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<64xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source[%idx0] into %out lane(%lane)
        : tensor<2048xf32>, vector<64xindex>, tensor<64xf32>, index -> tensor<64xf32>
    }
  }
  return %result : tensor<64xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_static
//       CHECK:   scf.forall
//       CHECK:     iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : memref<2048xf32, strided<[?], offset: ?>>, vector<64xindex>, memref<64xf32, strided<[?], offset: ?>>, index

// -----

// Test bufferization with different element types
func.func @bufferize_coalesced_gather_dma_f16(%idx0: vector<128xindex>,
                                               %source: tensor<8192xf16>,
                                               %dest: tensor<128xf16>,
                                               %lane: index) -> tensor<128xf16> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<128xf16>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source[%idx0] into %out lane(%lane)
        : tensor<8192xf16>, vector<128xindex>, tensor<128xf16>, index -> tensor<128xf16>
    }
  }
  return %result : tensor<128xf16>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_f16
//       CHECK:   scf.forall
//       CHECK:     iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : memref<8192xf16, strided<[?], offset: ?>>, vector<128xindex>, memref<128xf16, strided<[?], offset: ?>>, index

// -----

// Test bufferization with 1D tensors
func.func @bufferize_coalesced_gather_dma_1d(%indices: vector<1024xindex>,
                                              %source: tensor<2048xf32>,
                                              %dest: tensor<1024xf32> {bufferization.writable = true},
                                              %lane: index) -> tensor<1024xf32> {
  %c32 = arith.constant 32 : index
  %result = scf.forall (%i) in (%c32) shared_outs(%out = %dest) -> (tensor<1024xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source[%indices] into %out lane(%lane)
        : tensor<2048xf32>, vector<1024xindex>, tensor<1024xf32>, index -> tensor<1024xf32>
    }
  }
  return %result : tensor<1024xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_1d
//       CHECK:   %[[C32:.+]] = arith.constant 32 : index
//       CHECK:   scf.forall (%{{.+}}) in (%[[C32]])
//       CHECK:     iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : memref<2048xf32, strided<[?], offset: ?>>, vector<1024xindex>, memref<1024xf32, strided<[?], offset: ?>>, index

// -----

// Test bufferization in nested forall loops (workgroup and subgroup levels)
func.func @bufferize_coalesced_gather_dma_nested(%idx0: vector<8xindex>,
                                                  %source: tensor<2048xf32>,
                                                  %dest: tensor<8xf32> {bufferization.writable = true},
                                                  %lane: index) -> tensor<8xf32> {
  %result = scf.forall (%wg_i, %wg_j) in (16, 1) shared_outs(%wg_out = %dest) -> (tensor<8xf32>) {
    %inner_result = scf.forall (%sg_i, %sg_j) in (32, 1) shared_outs(%sg_out = %wg_out) -> (tensor<8xf32>) {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %source[%idx0] into %sg_out lane(%lane)
          : tensor<2048xf32>, vector<8xindex>, tensor<8xf32>, index -> tensor<8xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner_result into %wg_out[0] [8] [1]
        : tensor<8xf32> into tensor<8xf32>
    }
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  return %result : tensor<8xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_nested
//       CHECK:   scf.forall (%{{.+}}, %{{.+}}) in (16, 1)
//       CHECK:     scf.forall (%{{.+}}, %{{.+}}) in (32, 1)
//       CHECK:       iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : memref<2048xf32, strided<[?], offset: ?>>, vector<8xindex>, memref<8xf32, strided<[?], offset: ?>>, index

// -----

// Test bufferization without indices (contiguous gather)
func.func @bufferize_coalesced_gather_dma_no_indices(%source: tensor<1x32xf32>,
                                                      %dest: tensor<1x32xf32>,
                                                      %lane: index) -> tensor<1x32xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<1x32xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source into %out lane(%lane)
        : tensor<1x32xf32>, tensor<1x32xf32>, index -> tensor<1x32xf32>
    }
  }
  return %result : tensor<1x32xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_no_indices
//       CHECK:   scf.forall
//       CHECK:     iree_gpu.coalesced_gather_dma %{{.+}} into %{{.+}} lane(%{{.+}}) : memref<1x32xf32, strided<[?, ?], offset: ?>>, memref<1x32xf32, strided<[?, ?], offset: ?>>, index
