// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma))" %s --split-input-file | FileCheck %s

#config = #iree_gpu.lowering_config<{subgroup = [8, 1]}>

// Test that the GPUConvertToCoalescedDMA pass converts iree_linalg_ext.gather to tiled scf.forall with coalesced_gather_dma
// CHECK-LABEL: func.func @gather_tile_to_subgroup
func.func @gather_tile_to_subgroup(%source: tensor<1024xf32>, %indices: tensor<32x16xi32>, %init: tensor<32x16xf32>) -> tensor<32x16xf32> {
  // CHECK: %[[RESULT:.*]] = scf.forall (%[[IV0:.*]], %[[IV1:.*]]) = (0, 0) to (32, 16) step (8, 1)
  // CHECK-SAME: shared_outs(%[[OUT:.*]] = %{{.*}}) -> (tensor<32x16xf32>)
  %result = iree_linalg_ext.gather {lowering_config = #config}
    dimension_map = [0]
    ins(%source, %indices : tensor<1024xf32>, tensor<32x16xi32>)
    outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>

  // CHECK: %[[INDICES_SLICE:.*]] = tensor.extract_slice %{{.*}}[%[[IV0]], %[[IV1]]] [8, 1] [1, 1]
  // CHECK-SAME: tensor<32x16xi32> to tensor<8x1xi32>

  // CHECK: %[[INDEX_CAST:.*]] = arith.index_cast %[[INDICES_SLICE]]
  // CHECK-SAME: tensor<8x1xi32> to tensor<8x1xindex>

  // CHECK: scf.forall.in_parallel {
  // CHECK:   iree_gpu.coalesced_gather_dma %[[INDEX_CAST]], %{{.*}} into %[[OUT]]
  // CHECK-SAME: tensor<8x1xindex>, tensor<1024xf32>, tensor<32x16xf32> -> tensor<32x16xf32>
  // CHECK: }
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  // CHECK-NOT: iree_linalg_ext.gather
  return %result : tensor<32x16xf32>
}

// -----

#config2 = #iree_gpu.lowering_config<{subgroup = [4, 4]}>

// Test with different tile sizes
// CHECK-LABEL: func.func @gather_tile_4x4
func.func @gather_tile_4x4(%source: tensor<256xf32>, %indices: tensor<16x16xi32>, %init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  // CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (16, 16) step (4, 4)
  // CHECK-SAME: shared_outs(%[[OUT:.*]] = %{{.*}}) -> (tensor<16x16xf32>)
  %result = iree_linalg_ext.gather {lowering_config = #config2}
    dimension_map = [0]
    ins(%source, %indices : tensor<256xf32>, tensor<16x16xi32>)
    outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>

  // CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [4, 4] [1, 1]
  // CHECK-SAME: tensor<16x16xi32> to tensor<4x4xi32>

  // CHECK: arith.index_cast %{{.*}} : tensor<4x4xi32> to tensor<4x4xindex>

  // CHECK: scf.forall.in_parallel {
  // CHECK:   iree_gpu.coalesced_gather_dma %{{.*}}, %{{.*}} into %[[OUT]]
  // CHECK-SAME: tensor<4x4xindex>, tensor<256xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
  // CHECK: }
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  return %result : tensor<16x16xf32>
}
