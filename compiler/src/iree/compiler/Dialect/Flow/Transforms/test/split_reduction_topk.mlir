// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-flow-split-reduction-ops))" %s | FileCheck %s

!typef32 = tensor<4x8000xf32>
!otypef32 = tensor<4x40xf32>
!otypei32 = tensor<4x40xi32>

func.func @topk2d(
  %input_values: !typef32,
  %out_values: !otypef32, 
  %out_indices: !otypei32) -> (!otypef32, !otypei32) {
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values: !typef32)
        outs(%out_values, %out_indices : !otypef32, !otypei32) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> !otypef32, !otypei32
  return %0#0, %0#1 : !otypef32, !otypei32
}

//      CHECK-LABEL: func.func @topk2d
//      CHECK: iree_linalg_ext.topk dimension(2) ins(%{{.*}} : tensor<4x16x500xf32>) outs(%{{.*}}, %{{.*}} : tensor<4x16x40xf32>, tensor<4x16x40xi32>)
//      CHECK: linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%{{.*}}#1 : tensor<4x16x40xi32>)
//      CHECK: iree_linalg_ext.topk dimension(2) ins(%{{.*}}, %{{.*}} : tensor<4x5x128xf32>, tensor<4x5x128xi32>) outs(%{{.*}}, %{{.*}} : tensor<4x5x40xf32>, tensor<4x5x40xi32>)
//      CHECK: iree_linalg_ext.topk dimension(1) ins(%{{.*}}, %{{.*}} : tensor<4x200xf32>, tensor<4x200xi32>) outs(%{{.*}}, %{{.*}} : tensor<4x40xf32>, tensor<4x40xi32>) {