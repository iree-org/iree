// RUN: iree-opt --iree-flow-interchange-generic-ops %s | FileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1)>
//      CHECK: util.func public @interchange
//      CHECK:   linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
util.func public @interchange(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>],
    iterator_types = ["reduction", "parallel", "parallel", "parallel"]}
  ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  outs(%arg2 : tensor<?x?x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %m = arith.mulf %arg3, %arg4 : f32
    %a = arith.addf %arg5, %m : f32
    linalg.yield %a : f32
  } -> tensor<?x?x?xf32>
  util.return %0 : tensor<?x?x?xf32>
}
