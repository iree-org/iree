// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(util.func(iree-flow-form-dispatch-regions{aggressive-fusion=true}, iree-flow-clone-producers-into-dispatch-regions, iree-flow-convert-dispatch-regions-to-workgroups), cse, canonicalize, cse)" %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @linalgext_scatter_fusion() -> tensor<8192x16x8x128xf32> {
  %0 = tensor.empty() : tensor<4x1xi32>
  %1 = tensor.empty() : tensor<4x1xi64>
  %2 = tensor.empty() : tensor<4x1x16x8x128xf32>
  %3 = tensor.empty() : tensor<4x1x16x8x128xf32>
  %4 = tensor.empty() : tensor<8192x16x8x128xf32>
  %5 = tensor.empty() : tensor<8192x16x8x128xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<4x1xi64>) outs(%0 : tensor<4x1xi32>) {
  ^bb0(%in: i64, %out: i32):
    %10 = arith.trunci %in : i64 to i32
    linalg.yield %10 : i32
  } -> tensor<4x1xi32>

  %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<4x1x16x8x128xf32>) outs(%3 : tensor<4x1x16x8x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %10 = arith.addf %in, %out : f32
    linalg.yield %10 : f32
  } -> tensor<4x1x16x8x128xf32>

  %8 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%7, %6 : tensor<4x1x16x8x128xf32>, tensor<4x1xi32>) outs(%4 : tensor<8192x16x8x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    iree_linalg_ext.yield %arg0 : f32
  } -> tensor<8192x16x8x128xf32>

  // Dont fuse with scatter's consumer
  %9 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<8192x16x8x128xf32>) outs(%5 : tensor<8192x16x8x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %10 = arith.addf %in, %out : f32
    linalg.yield %10 : f32
  } -> tensor<8192x16x8x128xf32>
  util.return %9 : tensor<8192x16x8x128xf32>
}

// CHECK:     util.func public @linalgext_scatter_fusion
// CHECK:       %[[RESULT:.+]] = flow.dispatch.workgroups
// CHECK:         %[[INDICES:.+]] = linalg.generic
// CHECK:         %[[UPDATE:.+]] = linalg.generic
// CHECK:         %[[SCATTER_RESULT:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:      ins(%[[UPDATE]], %[[INDICES]] : tensor<4x1x16x8x128xf32>, tensor<4x1xi32>)
// CHECK:       flow.dispatch.workgroups
// CHECK:         %[[GEN2:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[INPUT:.+]] : tensor<8192x16x8x128xf32>)



// -----


#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @linalgext_reverse_fusion() -> tensor<10x10xi32> {
  %0 = tensor.empty() : tensor<10x10xi64>
  %1 = tensor.empty() : tensor<10x10xi32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<10x10xi64>) outs(%1 : tensor<10x10xi32>) {
  ^bb0(%in: i64, %out: i32):
    %7 = arith.trunci %in : i64 to i32
    linalg.yield %7 : i32
  } -> tensor<10x10xi32>
  %3 = tensor.empty() : tensor<10x10xi32>
  %4 = iree_linalg_ext.reverse dimensions(dense<0> : tensor<1xi64>) ins(%2 : tensor<10x10xi32>) outs(%3 : tensor<10x10xi32>) : tensor<10x10xi32>

  // dont fuse with with reverse's consumer
  %5 = tensor.empty() : tensor<10x10xi32>
  %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<10x10xi32>) outs(%5 : tensor<10x10xi32>) {
  ^bb0(%in: i32, %out: i32):
    %7 = arith.addi %in, %out : i32
    linalg.yield %7 : i32
  } -> tensor<10x10xi32>
  util.return %6 : tensor<10x10xi32>
}

// CHECK:     util.func public @linalgext_reverse_fusion
// CHECK:       flow.dispatch.workgroups
// CHECK:       %[[SHRUNK:.+]] = linalg.generic
// CHECK:       %[[REVERSED:.+]] = iree_linalg_ext.reverse
// CHECK:         ins(%[[SHRUNK]] : tensor<10x10xi32>)
// CHECK:       flow.dispatch.workgroups
// CHECK:       %[[GEN:.+]] = linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @linalgext_scatter_fusion() -> tensor<8192x16x8x128xf32> {
  %6 = tensor.empty() : tensor<4x1xi32>
  %2 = tensor.empty() : tensor<4x1x16x8x128xf32>
  %4 = tensor.empty() : tensor<10x8192x16x8x128xf32>

  %outs = tensor.extract_slice %4[0, 0, 0, 0, 0][1, 8192, 16, 8, 128][1, 1, 1, 1, 1] :
    tensor<10x8192x16x8x128xf32> to tensor<8192x16x8x128xf32>

  %8 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
    ins(%2, %6 : tensor<4x1x16x8x128xf32>, tensor<4x1xi32>)
    outs(%outs : tensor<8192x16x8x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    iree_linalg_ext.yield %arg0 : f32
  } -> tensor<8192x16x8x128xf32>

  util.return %8 : tensor<8192x16x8x128xf32>
}

// CHECK:     util.func public @linalgext_scatter_fusion
// CHECK:       %[[RESULT:.+]] = flow.dispatch.workgroups
// CHECK:         %[[OUTS:.+]] = tensor.extract_slice
// CHECK:         %[[SCATTER_RESULT:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:      outs(%[[OUTS]] : tensor<8192x16x8x128xf32>)
