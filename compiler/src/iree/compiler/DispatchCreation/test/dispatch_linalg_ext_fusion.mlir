// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}, iree-dispatch-creation-clone-producers-into-dispatch-regions), cse, canonicalize, cse)" %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @linalgext_scatter_dispatch() -> tensor<8192x16x8x128xf32> {
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

// CHECK-LABEL:     util.func public @linalgext_scatter_dispatch
//       CHECK:       %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:         %[[INDICES:.+]] = linalg.generic
//       CHECK:         %[[UPDATE:.+]] = linalg.generic
//       CHECK:         %[[SCATTER_RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:           ins(%[[UPDATE]], %[[INDICES]] : tensor<4x1x16x8x128xf32>, tensor<4x1xi32>)
//       CHECK:         flow.return %[[SCATTER_RESULT]]
//       CHECK:       flow.dispatch.region
//       CHECK:         %[[GEN2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[INPUT:.+]] : tensor<8192x16x8x128xf32>)
//       CHECK:         flow.return %[[GEN2]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @linalgext_scatter_clone() -> tensor<8192x16x8x128xf32> {
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

// CHECK-LABEL:     util.func public @linalgext_scatter_clone
//       CHECK:       %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:         %[[OUTS:.+]] = tensor.extract_slice
//       CHECK:         %[[SCATTER_RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:           outs(%[[OUTS]] : tensor<8192x16x8x128xf32>)
//       CHECK:         flow.return %[[SCATTER_RESULT]]

// -----

util.func public @attention_dispatch(%arg0: tensor<?x?x?xf16>, %arg1: tensor<?x?x?xf16>, %arg2: tensor<?x?x?xf16>, %arg3: f16, %arg4: tensor<?x?x?xf16>, %arg5: tensor<?x?x?xf16>, %arg6: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>

  %3 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%0, %1, %2, %arg3 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f16) outs(%arg4 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<?x?x?xf16>) outs(%arg4 : tensor<?x?x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.mulf %in, %in : f16
    linalg.yield %5 : f16
  } -> tensor<?x?x?xf16>
  util.return %4 : tensor<?x?x?xf16>
}

// CHECK-LABEL:     util.func public @attention_dispatch
//       CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN0:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN0]]
//       CHECK:       %[[DISPATCH1:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN1:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN1]]
//       CHECK:       %[[DISPATCH2:.+]] = flow.dispatch.region
//  CHECK-NEXT:         %[[GEN2:.+]] = linalg.generic
//       CHECK:         flow.return %[[GEN2]]
//       CHECK:       %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:         %[[ATTN:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:           ins(%[[DISPATCH0]], %[[DISPATCH1]], %[[DISPATCH2]]
//       CHECK:         %[[GEN2:.+]] = linalg.generic
//  CHECK-SAME:           ins(%[[ATTN]]
//       CHECK:         flow.return %[[GEN2]]
