// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma))" %s --split-input-file | FileCheck %s

#config = #iree_gpu.lowering_config<{subgroup = [1, 32], thread = [1, 1]}>

// CHECK-LABEL: func.func @gather_tile_to_subgroup
func.func @gather_tile_to_subgroup(%source: tensor<1024xf32>, %indices: tensor<32x32xindex>, %init: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %result = iree_linalg_ext.gather {lowering_config = #config}
    dimension_map = [0]
    ins(%source, %indices : tensor<1024xf32>, tensor<32x32xindex>)
    outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>

  // CHECK: %[[OUTER:.*]] = scf.forall (%[[WG_I:.*]], %[[WG_J:.*]]) = (0, 0) to (32, 32) step (1, 32)
  // CHECK-SAME: shared_outs(%[[WG_OUT:.*]] = %{{.*}}) -> (tensor<32x32xf32>)

  // CHECK: %[[INDICES_WG_SLICE:.*]] = tensor.extract_slice %{{.*}}[%[[WG_I]], %[[WG_J]]] [1, 32] [1, 1]
  // CHECK-SAME: tensor<32x32xindex> to tensor<1x32xindex>

  // CHECK: %[[DEST_WG_SLICE:.*]] = tensor.extract_slice %[[WG_OUT]][%[[WG_I]], %[[WG_J]]] [1, 32] [1, 1]
  // CHECK-SAME: tensor<32x32xf32> to tensor<1x32xf32>

  // CHECK: %[[INNER:.*]] = scf.forall (%[[TH_I:.*]], %[[TH_J:.*]]) in (1, 32)
  // CHECK-SAME: shared_outs(%[[TH_OUT:.*]] = %[[DEST_WG_SLICE]]) -> (tensor<1x32xf32>)

  // CHECK: scf.forall.in_parallel {
  // CHECK:   iree_gpu.coalesced_gather_dma %[[INDICES_WG_SLICE]], %{{.*}} into %[[TH_OUT]]
  // CHECK-SAME: tensor<1x32xindex>, tensor<1024xf32>, tensor<1x32xf32> -> tensor<1x32xf32>
  // CHECK: }
  // CHECK: } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

  // CHECK: scf.forall.in_parallel {
  // CHECK:   tensor.parallel_insert_slice %[[INNER]] into %[[WG_OUT]][%[[WG_I]], %[[WG_J]]] [1, 32] [1, 1]
  // CHECK: }
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  // CHECK-NOT: iree_linalg_ext.gather
  return %result : tensor<32x32xf32>
}
