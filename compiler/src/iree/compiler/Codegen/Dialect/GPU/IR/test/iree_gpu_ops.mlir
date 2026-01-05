// RUN: iree-opt %s --split-input-file --verify-diagnostics | FileCheck %s

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

// Test basic coalesced_gather_dma with static shapes.
// One 1D index for a 1D source gathering into a 1D result.
func.func @coalesced_gather_dma_static(%idx0: vector<64xi32>, %source: tensor<4096xf32>, %dest: tensor<64xf32>, %lane: index) -> tensor<64xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<64xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source[%idx0] into %out lane(%lane)
        : tensor<4096xf32>, vector<64xi32>, tensor<64xf32>, index -> tensor<64xf32>
    }
  }
  return %result : tensor<64xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_static
//       CHECK:   scf.forall
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : tensor<4096xf32>, vector<64xi32>, tensor<64xf32>, index -> tensor<64xf32>

// -----

// Test coalesced_gather_dma with different element types.
func.func @coalesced_gather_dma_f16(%idx0: vector<128xi32>, %source: tensor<8192xf16>, %dest: tensor<128xf16>, %lane: index) -> tensor<128xf16> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<128xf16>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source[%idx0] into %out lane(%lane)
        : tensor<8192xf16>, vector<128xi32>, tensor<128xf16>, index -> tensor<128xf16>
    }
  }
  return %result : tensor<128xf16>
}

// CHECK-LABEL: func @coalesced_gather_dma_f16
//       CHECK:   scf.forall
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : tensor<8192xf16>, vector<128xi32>, tensor<128xf16>, index -> tensor<128xf16>

// -----

func.func @coalesced_gather_dma_1d(%indices: vector<1024xi32>, %source: tensor<2048xf32>, %dest: tensor<1024xf32>, %lane: index) -> tensor<1024xf32> {
  %c32 = arith.constant 32 : index
  %result = scf.forall (%i) in (%c32) shared_outs(%out = %dest) -> (tensor<1024xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source[%indices] into %out lane(%lane)
        : tensor<2048xf32>, vector<1024xi32>, tensor<1024xf32>, index -> tensor<1024xf32>
    }
  }
  return %result : tensor<1024xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_1d
//       CHECK:   %[[C32:.+]] = arith.constant 32 : index
//       CHECK:   scf.forall (%{{.+}}) in (%[[C32]])
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : tensor<2048xf32>, vector<1024xi32>, tensor<1024xf32>, index -> tensor<1024xf32>

// -----

func.func @coalesced_gather_dma_copy(%source: tensor<32x128xf32>, %dest: tensor<32x128xf32>, %lane: index) -> tensor<32x128xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<32x128xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source into %out lane(%lane)
        : tensor<32x128xf32>, tensor<32x128xf32>, index -> tensor<32x128xf32>
    }
  }
  return %result : tensor<32x128xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_copy
//       CHECK:   scf.forall
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %{{.+}} into %{{.+}} lane(%{{.+}}) : tensor<32x128xf32>, tensor<32x128xf32>, index -> tensor<32x128xf32>

// -----

func.func @coalesced_gather_dma_copy_memref(%source: memref<1x32xf32, strided<[128, 1], offset: ?>>, %dest: memref<1x32xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, %lane: index) {
  iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : memref<1x32xf32, strided<[128, 1], offset: ?>>, memref<1x32xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, index
  return
}

// CHECK-LABEL: func @coalesced_gather_dma_copy_memref
//       CHECK:   iree_gpu.coalesced_gather_dma %{{.+}} into %{{.+}} lane(%{{.+}}) : memref<1x32xf32, strided<[128, 1], offset: ?>>, memref<1x32xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, index

// -----

func.func @coalesced_gather_dma_tensor_indices(%idx0: tensor<64xi32>, %source: tensor<4096xf32>, %dest: tensor<64xf32>, %lane: index) -> tensor<64xf32> {
  %c1 = arith.constant 1 : index
  %result = scf.forall (%i) in (%c1) shared_outs(%out = %dest) -> (tensor<64xf32>) {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %source[%idx0] into %out lane(%lane)
        : tensor<4096xf32>, tensor<64xi32>, tensor<64xf32>, index -> tensor<64xf32>
    }
  }
  return %result : tensor<64xf32>
}

// CHECK-LABEL: func @coalesced_gather_dma_tensor_indices
//       CHECK:   scf.forall
//       CHECK:     scf.forall.in_parallel
//       CHECK:       iree_gpu.coalesced_gather_dma %{{.+}}[%{{.+}}] into %{{.+}} lane(%{{.+}}) : tensor<4096xf32>, tensor<64xi32>, tensor<64xf32>, index -> tensor<64xf32>

