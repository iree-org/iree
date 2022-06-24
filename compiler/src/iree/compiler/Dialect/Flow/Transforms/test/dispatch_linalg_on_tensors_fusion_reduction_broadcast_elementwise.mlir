// RUN: iree-opt --iree-flow-fuse-reduction-broadcast-elementwise --split-input-file --pass-pipeline="func.func(iree-flow-dispatch-linalg-on-tensors-pass)" --canonicalize -cse %s | FileCheck %s
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reduction_broadcast_elementwise(%a: tensor<12x16x16xf32>, %b: tensor<12x16x16xf32>) -> tensor<12x16x16xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = linalg.init_tensor [12, 16] : tensor<12x16xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<12x16xf32>) -> tensor<12x16xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a : tensor<12x16x16xf32>) outs(%38 : tensor<12x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %780 = arith.maxf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16xf32>
  %40 = linalg.init_tensor [12, 16, 16] : tensor<12x16x16xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%b, %39 : tensor<12x16x16xf32>, tensor<12x16xf32>) outs(%40 : tensor<12x16x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16x16xf32>
  return %42 : tensor<12x16x16xf32>
}

// Check that two generic ops are dispatched together.
// The first generic (reduction) is directly used by the second generic (elementwise).

// CHECK-LABEL: func.func @reduction_broadcast_elementwise
//      CHECK: flow.dispatch.workgroups
//      CHECK:   %[[RED:.+]] = linalg.generic
//      CHECK:   linalg.generic
//      CHECK-SAME: %[[RED]]
