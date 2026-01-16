// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-test-convert-forall-to-generic-nest))" --split-input-file | FileCheck %s

// Test converting scf.forall with gpu.thread mapping to nested pcf.generic
// with subgroup scope (outer) and lane scope (inner).

func.func @test_1d_thread_mapping(%init: tensor<64xf32>) -> tensor<64xf32> {
  %result = scf.forall (%i) in (64) shared_outs(%out = %init) -> tensor<64xf32> {
    %slice = tensor.extract_slice %out[%i] [1] [1] : tensor<64xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i] [1] [1] : tensor<1xf32> into tensor<64xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return %result : tensor<64xf32>
}

// CHECK-LABEL: func.func @test_1d_thread_mapping
//  CHECK-SAME:   %[[INIT:.+]]: tensor<64xf32>
//       CHECK: %[[RESULT:.+]] = pcf.generic
//  CHECK-SAME:   scope(#iree_gpu.subgroup_scope)
//       CHECK:   execute(%[[REF:.+]] = %[[INIT]])[%[[SUBGROUP_ID:.+]]: index, %[[NUM_SUBGROUPS:.+]]: index]
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.lane_scope)
//       CHECK:     execute[%[[LANE_ID:.+]]: index, %[[SUBGROUP_SIZE:.+]]: index]
//       CHECK:     %[[LIN_ID:.+]] = affine.linearize_index [%[[SUBGROUP_ID]], %[[LANE_ID]]] by (%[[NUM_SUBGROUPS]], %[[SUBGROUP_SIZE]])
//       CHECK:     %[[TOTAL_COUNT:.+]] = arith.muli %[[NUM_SUBGROUPS]], %[[SUBGROUP_SIZE]]
//       CHECK:     %[[TILE_SIZE:.+]] = arith.ceildivui %{{.+}}, %[[TOTAL_COUNT]]
//       CHECK:     %[[START:.+]] = arith.muli %[[LIN_ID]], %[[TILE_SIZE]]
//       CHECK:     %[[END_UNCLAMPED:.+]] = arith.addi %[[START]], %[[TILE_SIZE]]
//       CHECK:     %[[END:.+]] = arith.minui %[[END_UNCLAMPED]]
//       CHECK:     scf.forall (%[[IV:.+]]) = (%[[START]]) to (%[[END]])
//       CHECK:       pcf.write_slice %{{.+}} into %[[REF]][%[[IV]]]
//       CHECK:     pcf.return
//       CHECK:   pcf.return
//       CHECK: return %[[RESULT]]

// -----

func.func @test_2d_thread_mapping(%init: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %result = scf.forall (%i, %j) in (64, 128) shared_outs(%out = %init) -> tensor<64x128xf32> {
    %slice = tensor.extract_slice %out[%i, %j] [1, 1] [1, 1] : tensor<64x128xf32> to tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i, %j] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<64x128xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return %result : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_2d_thread_mapping
//  CHECK-SAME:   %[[INIT:.+]]: tensor<64x128xf32>
//       CHECK: pcf.generic
//  CHECK-SAME:   scope(#iree_gpu.subgroup_scope)
//       CHECK:   execute(%[[REF:.+]] = %[[INIT]])[%[[SUBGROUP_ID:.+]]: index, %[[NUM_SUBGROUPS:.+]]: index]
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.lane_scope)
//       CHECK:     execute[%[[LANE_ID:.+]]: index, %[[SUBGROUP_SIZE:.+]]: index]
//       CHECK:     %[[LIN_ID:.+]] = affine.linearize_index [%[[SUBGROUP_ID]], %[[LANE_ID]]] by (%[[NUM_SUBGROUPS]], %[[SUBGROUP_SIZE]])
//       CHECK:     %[[TOTAL_COUNT:.+]] = arith.muli %[[NUM_SUBGROUPS]], %[[SUBGROUP_SIZE]]
//       CHECK:     scf.forall (%[[IV:.+]]) =
//       CHECK:       %[[INDICES:.+]]:2 = affine.delinearize_index %[[IV]] into (64, 128)
//       CHECK:       pcf.write_slice %{{.+}} into %[[REF]][%[[INDICES]]#0, %[[INDICES]]#1]
//       CHECK:     pcf.return
//       CHECK:   pcf.return
