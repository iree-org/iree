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
func.func @coalesced_gather_dma_static(%indices: tensor<64x32xi32>, %source: tensor<1024x64xf32>, %dest: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<64x32xf32>) {
    %gathered = iree_gpu.coalesced_gather_dma %indices, %source into %out
      : tensor<64x32xi32>, tensor<1024x64xf32>, tensor<64x32xf32> -> tensor<64x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %gathered into %out[0, 0] [64, 32] [1, 1]
        : tensor<64x32xf32> into tensor<64x32xf32>
    }
  }
  return %result : tensor<64x32xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_static
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<64x32xi32>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<1024x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<64x32xf32>
//       CHECK:   scf.forall
//       CHECK:     %[[GATHERED:.+]] = iree_gpu.coalesced_gather_dma %[[INDICES]], %[[SOURCE]] into %{{.+}} : tensor<64x32xi32>, tensor<1024x64xf32>, tensor<64x32xf32> -> tensor<64x32xf32>
//       CHECK:     scf.forall.in_parallel

// -----

// Test coalesced_gather_dma with dynamic shapes
func.func @coalesced_gather_dma_dynamic(%indices: tensor<?x?xi32>, %source: tensor<?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %dest, %c0 : tensor<?x?xf32>
  %c1_dim = arith.constant 1 : index
  %d1 = tensor.dim %dest, %c1_dim : tensor<?x?xf32>
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<?x?xf32>) {
    %gathered = iree_gpu.coalesced_gather_dma %indices, %source into %out
      : tensor<?x?xi32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %gathered into %out[0, 0] [%d0, %d1] [1, 1]
        : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_dynamic
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<?x?xf32>
//       CHECK:   scf.forall
//       CHECK:     %[[GATHERED:.+]] = iree_gpu.coalesced_gather_dma %[[INDICES]], %[[SOURCE]] into %{{.+}} : tensor<?x?xi32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
//       CHECK:     scf.forall.in_parallel

// -----

// Test coalesced_gather_dma with f16 data type (verifying size constraints)
// indices: 128x64 = 8192 elements, dest: 128x64xf16 = 8192 * 2 bytes = 16384 bytes, ratio = 16384/8192 = 2
func.func @coalesced_gather_dma_f16(%indices: tensor<128x64xi32>, %source: tensor<2048x64xf16>, %dest: tensor<128x64xf16>) -> tensor<128x64xf16> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<128x64xf16>) {
    %gathered = iree_gpu.coalesced_gather_dma %indices, %source into %out
      : tensor<128x64xi32>, tensor<2048x64xf16>, tensor<128x64xf16> -> tensor<128x64xf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %gathered into %out[0, 0] [128, 64] [1, 1]
        : tensor<128x64xf16> into tensor<128x64xf16>
    }
  }
  return %result : tensor<128x64xf16>
}

// CHECK-LABEL: func @coalesced_gather_dma_f16
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<128x64xi32>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<2048x64xf16>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<128x64xf16>
//       CHECK:   scf.forall
//       CHECK:     %[[GATHERED:.+]] = iree_gpu.coalesced_gather_dma %[[INDICES]], %[[SOURCE]] into %{{.+}} : tensor<128x64xi32>, tensor<2048x64xf16>, tensor<128x64xf16> -> tensor<128x64xf16>
//       CHECK:     scf.forall.in_parallel

// -----

// Test coalesced_gather_dma with 1D tensors
// indices: 1024 elements, dest: 256xf32 = 256 * 4 bytes = 1024 bytes, ratio = 1024/1024 = 1
func.func @coalesced_gather_dma_1d(%indices: tensor<1024xi32>, %source: tensor<1024xf32>, %dest: tensor<256xf32>) -> tensor<256xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<256xf32>) {
    %gathered = iree_gpu.coalesced_gather_dma %indices, %source into %out
      : tensor<1024xi32>, tensor<1024xf32>, tensor<256xf32> -> tensor<256xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %gathered into %out[0] [256] [1]
        : tensor<256xf32> into tensor<256xf32>
    }
  }
  return %result : tensor<256xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_1d
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<1024xi32>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<1024xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<256xf32>
//       CHECK:   scf.forall
//       CHECK:     %[[GATHERED:.+]] = iree_gpu.coalesced_gather_dma %[[INDICES]], %[[SOURCE]] into %{{.+}} : tensor<1024xi32>, tensor<1024xf32>, tensor<256xf32> -> tensor<256xf32>
//       CHECK:     scf.forall.in_parallel

// -----

// Test coalesced_gather_dma within scf.forall with tensor.parallel_insert_slice
// Using simpler dimensions to satisfy the verifier constraint
func.func @coalesced_gather_dma_in_forall(%indices: tensor<1024x64xi32>, %source: tensor<2048x64xf32>, %dest: tensor<256x16xf32>) -> tensor<256x16xf32> {
  %c4 = arith.constant 4 : index
  %result = scf.forall (%i) in (%c4) shared_outs(%out = %dest) -> (tensor<256x16xf32>) {
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c4_idx = arith.constant 4 : index
    %offset_idx = arith.muli %i, %c256 : index
    %offset_dest = arith.muli %i, %c4_idx : index

    // Extract slices for this iteration
    // indices: 64x16 = 1024 elements, dest: 64x16xf32 = 1024 * 4 bytes = 4096 bytes, ratio = 4096/1024 = 4
    %indices_slice = tensor.extract_slice %indices[%offset_dest, 0] [64, 16] [1, 1]
      : tensor<1024x64xi32> to tensor<64x16xi32>
    %dest_slice = tensor.extract_slice %out[%offset_dest, 0] [64, 16] [1, 1]
      : tensor<256x16xf32> to tensor<64x16xf32>

    // Perform the gather DMA operation
    %gathered = iree_gpu.coalesced_gather_dma %indices_slice, %source into %dest_slice
      : tensor<64x16xi32>, tensor<2048x64xf32>, tensor<64x16xf32> -> tensor<64x16xf32>

    // Insert the result back into the output tensor
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %gathered into %out[%offset_dest, 0] [64, 16] [1, 1]
        : tensor<64x16xf32> into tensor<256x16xf32>
    }
  }
  return %result : tensor<256x16xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_in_forall
//  CHECK-SAME:   %[[INDICES:[A-Za-z0-9]+]]: tensor<1024x64xi32>
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: tensor<2048x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: tensor<256x16xf32>
//       CHECK:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[I:.+]]) in (%[[C4]]) shared_outs(%[[OUT:.+]] = %[[DEST]])
//       CHECK:     %[[INDICES_SLICE:.+]] = tensor.extract_slice %[[INDICES]]
//       CHECK:     %[[DEST_SLICE:.+]] = tensor.extract_slice %[[OUT]]
//       CHECK:     %[[GATHERED:.+]] = iree_gpu.coalesced_gather_dma %[[INDICES_SLICE]], %[[SOURCE]] into %[[DEST_SLICE]] : tensor<64x16xi32>, tensor<2048x64xf32>, tensor<64x16xf32> -> tensor<64x16xf32>
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[GATHERED]] into %[[OUT]]
//       CHECK:   return %[[RESULT]]
