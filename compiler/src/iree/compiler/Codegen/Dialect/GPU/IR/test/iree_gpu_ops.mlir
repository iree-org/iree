// RUN: iree-opt %s --split-input-file | FileCheck %s

func.func @barrier_region(%init: tensor<6x6xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.barrier_region ins(%init : tensor<6x6xf32>) {
  ^bb0(%intermediate: tensor<6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, 0] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @barrier_region
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9]+]]: tensor<6x6xf32>
//       CHECK:   iree_gpu.barrier_region ins(%[[INIT]] : tensor<6x6xf32>) {
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<6x6xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][0, 0] [3, 2] [1, 1]
//       CHECK:       iree_gpu.yield %[[SLICE]] : tensor<3x2xf32>
//       CHECK:   } : tensor<3x2xf32>

// -----

func.func @multi_result_barrier_region(%init: tensor<12x12xf32>) -> (tensor<2x1x3x2xf32>, index) {
  %0:2 = iree_gpu.barrier_region ins(%init : tensor<12x12xf32>) {
  ^bb0(%intermediate: tensor<12x12xf32>):
    %expand = tensor.expand_shape %intermediate [[0, 1], [2, 3]] output_shape [4, 3, 3, 4] : tensor<12x12xf32> into tensor<4x3x3x4xf32>
    %slice = tensor.extract_slice %expand[0, 0, 0, 0] [2, 1, 3, 2] [1, 1, 1, 1] : tensor<4x3x3x4xf32> to tensor<2x1x3x2xf32>
    %c0 = arith.constant 0 : index
    iree_gpu.yield %slice, %c0 : tensor<2x1x3x2xf32>, index
  } : tensor<2x1x3x2xf32>, index
  return %0#0, %0#1 : tensor<2x1x3x2xf32>, index
}

// CHECK-LABEL: func @multi_result_barrier_region
//       CHECK:   %{{.*}}:2 = iree_gpu.barrier_region ins(%{{.*}} : tensor<12x12xf32>)
//       CHECK:   } : tensor<2x1x3x2xf32>, index

// -----

func.func @multi_input_barrier_region(%x: index, %y: index) -> index {
  %0 = iree_gpu.barrier_region ins(%x, %y : index, index) {
  ^bb0(%ix: index, %iy: index):
    %sum = arith.addi %ix, %iy : index
    iree_gpu.yield %sum : index
  } : index
  return %0 : index
}

// CHECK-LABEL: func @multi_input_barrier_region
//       CHECK:   %{{.*}} = iree_gpu.barrier_region ins(%{{.*}}, %{{.*}} : index, index)
//       CHECK:   } : index

// -----

func.func @tensor_barrier(%input: tensor<?xf16>) -> tensor<?xf16> {
  %out = iree_gpu.value_barrier %input : tensor<?xf16>
  return %out : tensor<?xf16>
}

// CHECK-LABEL: func @tensor_barrier
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?xf16>
//       CHECK:   iree_gpu.value_barrier %[[INPUT]] : tensor<?xf16>

// -----

func.func @vector_barrier(%input: vector<8xf16>) -> vector<8xf16> {
  %out = iree_gpu.value_barrier %input : vector<8xf16>
  return %out : vector<8xf16>
}

// CHECK-LABEL: func @vector_barrier
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: vector<8xf16>
//       CHECK:   iree_gpu.value_barrier %[[INPUT]] : vector<8xf16>

// -----

func.func @vector_barrier_multiple_inputs(%input: vector<8xf16>) -> (vector<8xf16>, vector<8xf16>) {
  %out:2 = iree_gpu.value_barrier %input, %input : vector<8xf16>, vector<8xf16>
  return %out#0, %out#1 : vector<8xf16>, vector<8xf16>
}

// CHECK-LABEL: func @vector_barrier_multiple_inputs
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: vector<8xf16>
//       CHECK:   iree_gpu.value_barrier %[[INPUT]], %[[INPUT]] : vector<8xf16>, vector<8xf16>

// -----

func.func @tensor_barrier_multiple_inputs(%input: tensor<?xf16>) -> (tensor<?xf16>, tensor<?xf16>) {
  %out:2 = iree_gpu.value_barrier %input, %input : tensor<?xf16>, tensor<?xf16>
  return %out#0, %out#1 : tensor<?xf16>, tensor<?xf16>
}

// CHECK-LABEL: func @tensor_barrier_multiple_inputs
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?xf16>
//       CHECK:   iree_gpu.value_barrier %[[INPUT]], %[[INPUT]] : tensor<?xf16>, tensor<?xf16>

// -----

// Test basic coalesced_gather_dma with static shapes
// indices: 64x32 = 2048 elements, dest: 64x32xf32 = 2048 * 4 bytes = 8192 bytes, ratio = 8192/2048 = 4
func.func @coalesced_gather_dma_static(%indices: tensor<64x32xindex>, %source: tensor<1024x64xf32>, %dest: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<64x32xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %indices, %source into %out
        : tensor<64x32xindex>, tensor<1024x64xf32>, tensor<64x32xf32> -> tensor<64x32xf32>
    }
  }
  return %result : tensor<64x32xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_static
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<64x32xindex>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<1024x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<64x32xf32>
//       CHECK:   scf.forall
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %[[INDICES]], %[[SOURCE]] into %{{.+}} : tensor<64x32xindex>, tensor<1024x64xf32>, tensor<64x32xf32> -> tensor<64x32xf32>

// -----

// Test coalesced_gather_dma with dynamic shapes
func.func @coalesced_gather_dma_dynamic(%indices: tensor<?x?xindex>, %source: tensor<?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<?x?xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %indices, %source into %out
        : tensor<?x?xindex>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
    }
  }
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_dynamic
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<?x?xindex>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<?x?xf32>
//       CHECK:   scf.forall
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %[[INDICES]], %[[SOURCE]] into %{{.+}} : tensor<?x?xindex>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>

// -----

// Test coalesced_gather_dma with f16 data type (verifying size constraints)
// indices: 128x64 = 8192 elements, dest: 128x64xf16 = 8192 * 2 bytes = 16384 bytes, ratio = 16384/8192 = 2
func.func @coalesced_gather_dma_f16(%indices: tensor<128x64xindex>, %source: tensor<2048x64xf16>, %dest: tensor<128x64xf16>) -> tensor<128x64xf16> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<128x64xf16>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %indices, %source into %out
        : tensor<128x64xindex>, tensor<2048x64xf16>, tensor<128x64xf16> -> tensor<128x64xf16>
    }
  }
  return %result : tensor<128x64xf16>
}

// CHECK-LABEL: func @coalesced_gather_dma_f16
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<128x64xindex>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<2048x64xf16>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<128x64xf16>
//       CHECK:   scf.forall
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %[[INDICES]], %[[SOURCE]] into %{{.+}} : tensor<128x64xindex>, tensor<2048x64xf16>, tensor<128x64xf16> -> tensor<128x64xf16>

// -----

// Test coalesced_gather_dma with 1D tensors
func.func @coalesced_gather_dma_1d(%indices: tensor<1024xindex>, %source: tensor<1024xf32>, %dest: tensor<1024xf32>) -> tensor<1024xf32> {
  %c4 = arith.constant 4 : index
  %result = scf.forall (%i) in (%c4) shared_outs(%out = %dest) -> (tensor<1024xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %indices, %source into %out
        : tensor<1024xindex>, tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
    }
  }
  return %result : tensor<1024xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_1d
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<1024xindex>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<1024xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<1024xf32>
//       CHECK:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   scf.forall (%{{.+}}) in (%[[C4]])
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %[[INDICES]], %[[SOURCE]] into %{{.+}} : tensor<1024xindex>, tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>

// -----

// Test coalesced_gather_dma within 2-level nested scf.forall
func.func @coalesced_gather_dma_in_forall(%indices: tensor<1024x64xindex>, %source: tensor<2048x64xf32>, %dest: tensor<512x16xf32>) -> tensor<512x16xf32> {
  // Outer forall: warp level parallelism
  %result = scf.forall (%wg_i, %wg_j) in (2, 1) shared_outs(%wg_out = %dest) -> (tensor<512x16xf32>) {
    %c256 = arith.constant 256 : index
    %wg_offset = arith.muli %wg_i, %c256 : index

    %indices_wg_slice = tensor.extract_slice %indices[%wg_offset, 0] [256, 16] [1, 1]
      : tensor<1024x64xindex> to tensor<256x16xindex>

    %dest_wg_slice = tensor.extract_slice %wg_out[%wg_offset, 0] [256, 16] [1, 1]
      : tensor<512x16xf32> to tensor<256x16xf32>

    // Inner forall: thread level parallelism
    %inner_result = scf.forall (%sg_i) in (32) shared_outs(%sg_out = %dest_wg_slice) -> (tensor<256x16xf32>) {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %indices_wg_slice, %source into %sg_out
          : tensor<256x16xindex>, tensor<2048x64xf32>, tensor<256x16xf32> -> tensor<256x16xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner_result into %wg_out[%wg_offset, 0] [256, 16] [1, 1]
        : tensor<256x16xf32> into tensor<512x16xf32>
    }
  } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  return %result : tensor<512x16xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_in_forall
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<1024x64xindex>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<2048x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<512x16xf32>
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[WG_I:.+]], %[[WG_J:.+]]) in (2, 1) shared_outs(%[[WG_OUT:.+]] = %[[DEST]])
//       CHECK:     %[[WG_OFFSET:.+]] = arith.muli %[[WG_I]]
//       CHECK:     %[[INDICES_WG_SLICE:.+]] = tensor.extract_slice %[[INDICES]][%[[WG_OFFSET]], 0] [256, 16] [1, 1] : tensor<1024x64xindex> to tensor<256x16xindex>
//       CHECK:     %[[DEST_WG_SLICE:.+]] = tensor.extract_slice %[[WG_OUT]][%[[WG_OFFSET]], 0] [256, 16] [1, 1] : tensor<512x16xf32> to tensor<256x16xf32>
//       CHECK:     %[[INNER_RESULT:.+]] = scf.forall (%[[SG_I:.+]]) in (32) shared_outs(%[[SG_OUT:.+]] = %[[DEST_WG_SLICE]])
//       CHECK:       scf.forall.in_parallel
//       CHECK:         iree_gpu.coalesced_gather_dma %[[INDICES_WG_SLICE]], %[[SOURCE]] into %[[SG_OUT]] : tensor<256x16xindex>, tensor<2048x64xf32>, tensor<256x16xf32> -> tensor<256x16xf32>
//       CHECK:     } {mapping = [#gpu.thread<linear_dim_0>]}
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[INNER_RESULT]] into %[[WG_OUT]][%[[WG_OFFSET]], 0] [256, 16] [1, 1] : tensor<256x16xf32> into tensor<512x16xf32>
//       CHECK:   } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
//       CHECK:   return %[[RESULT]]
