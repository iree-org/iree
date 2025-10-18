// RUN: iree-opt %s --one-shot-bufferize="bufferize-function-boundaries" --split-input-file | FileCheck %s

// Test bufferization of coalesced_gather_dma with static shapes
func.func @bufferize_coalesced_gather_dma_static(%indices: tensor<64x32xindex>,
                                                  %source: tensor<1024x64xf32>,
                                                  %dest: tensor<64x32xf32> {bufferization.writable = true}) -> tensor<64x32xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<64x32xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %indices, %source into %out
        : tensor<64x32xindex>, tensor<1024x64xf32>, tensor<64x32xf32> -> tensor<64x32xf32>
    }
  }
  return %result : tensor<64x32xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_static
//       CHECK:   scf.forall
//       CHECK:     iree_gpu.coalesced_gather_dma %{{.+}}, %{{.+}} into %{{.+}} : memref<64x32xindex, strided<[?, ?], offset: ?>>, memref<1024x64xf32, strided<[?, ?], offset: ?>>, memref<64x32xf32, strided<[?, ?], offset: ?>> -> memref<64x32xf32, strided<[?, ?], offset: ?>>

// -----

// Test bufferization in nested forall loops (workgroup and subgroup levels)
func.func @bufferize_coalesced_gather_dma_nested(%indices: tensor<16x32xindex>,
                                                  %source: tensor<2048x64xf32>,
                                                  %dest: tensor<128x16xf32> {bufferization.writable = true}) -> tensor<128x16xf32> {
  %result = scf.forall (%wg_i, %wg_j) in (16, 1) shared_outs(%wg_out = %dest) -> (tensor<128x16xf32>) {
    %c8 = arith.constant 8 : index
    %wg_offset = arith.muli %wg_i, %c8 : index
    %indices_wg_slice = tensor.extract_slice %indices[%wg_offset, 0] [1, 32] [1, 1]
      : tensor<16x32xindex> to tensor<1x32xindex>
    %dest_wg_slice = tensor.extract_slice %wg_out[%wg_offset, 0] [8, 16] [1, 1]
      : tensor<128x16xf32> to tensor<8x16xf32>

    %inner_result = scf.forall (%sg_i, %sg_j) in (32, 1) shared_outs(%sg_out = %dest_wg_slice) -> (tensor<8x16xf32>) {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %indices_wg_slice, %source into %sg_out
          : tensor<1x32xindex>, tensor<2048x64xf32>, tensor<8x16xf32> -> tensor<8x16xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner_result into %wg_out[%wg_offset, 0] [8, 16] [1, 1]
        : tensor<8x16xf32> into tensor<128x16xf32>
    }
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  return %result : tensor<128x16xf32>
}

// CHECK-LABEL: func @bufferize_coalesced_gather_dma_nested
//       CHECK:   scf.forall (%{{.+}}, %{{.+}}) in (16, 1)
//       CHECK:     %{{.+}} = memref.subview
//       CHECK:     %{{.+}} = memref.subview
//       CHECK:     scf.forall (%{{.+}}, %{{.+}}) in (32, 1)
//       CHECK:       iree_gpu.coalesced_gather_dma %{{.+}}, %{{.+}} into %{{.+}} : memref<1x32xindex, strided<[?, ?], offset: ?>>, memref<2048x64xf32, strided<[?, ?], offset: ?>>, memref<8x16xf32, strided<[?, ?], offset: ?>> -> memref<8x16xf32, strided<[?, ?], offset: ?>>
