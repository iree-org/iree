// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-linalg-ext-test-reshape-fusion, canonicalize, cse, canonicalize))" %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @attention_static(%arg0: tensor<20x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16) -> tensor<2x10x4096x64xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<20x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16) outs(%0 : tensor<20x4096x64xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<20x4096x64xf16>
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape [2, 10, 4096, 64] : tensor<20x4096x64xf16> into tensor<2x10x4096x64xf16>
  util.return %expanded : tensor<2x10x4096x64xf16>
}

//CHECK-LABEL: func public @attention_static(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x10x4096x64xf16>
//  CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 4096, 16]
//  CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 1024, 16]
//  CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 1024, 64]
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       indexing_maps =
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
// CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[ATTENTION]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @attention_static_masked(%arg0: tensor<20x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16, %arg4: tensor<20x4096x1024xf16>) -> tensor<2x10x4096x64xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5]} ins(%arg0, %arg1, %arg2, %arg3, %arg4 : tensor<20x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16, tensor<20x4096x1024xf16>) outs(%0 : tensor<20x4096x64xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<20x4096x64xf16>
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape [2, 10, 4096, 64] : tensor<20x4096x64xf16> into tensor<2x10x4096x64xf16>
  util.return %expanded : tensor<2x10x4096x64xf16>
}

//CHECK-LABEL: func public @attention_static_masked(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<20x4096x1024xf16>)
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x10x4096x64xf16>
//  CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 4096, 16]
//  CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 1024, 16]
//  CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 1024, 64]
//  CHECK-DAG:   %[[MASK:.+]] = tensor.expand_shape %[[ARG4]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 4096, 1024]
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       indexing_maps =
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
// CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]], %[[MASK]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[ATTENTION]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @attention_expand_all(%arg0: tensor<20x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16) -> tensor<2x10x2048x2x2x32xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<20x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16) outs(%0 : tensor<20x4096x64xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<20x4096x64xf16>
  %expanded = tensor.expand_shape %1 [[0, 1], [2, 3], [4, 5]] output_shape [2, 10, 2048, 2, 2, 32] : tensor<20x4096x64xf16> into tensor<2x10x2048x2x2x32xf16>
  util.return %expanded : tensor<2x10x2048x2x2x32xf16>
}

//CHECK-LABEL: func public @attention_expand_all(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x10x2048x2x2x32xf16>
//  CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1], [2, 3], [4]] output_shape [2, 10, 2048, 2, 16] : tensor<20x4096x16xf16> into tensor<2x10x2048x2x16xf16>
//  CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1], [2], [3]] output_shape [2, 10, 1024, 16] : tensor<20x1024x16xf16> into tensor<2x10x1024x16xf16>
//  CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1], [2], [3, 4]] output_shape [2, 10, 1024, 2, 32] : tensor<20x1024x64xf16> into tensor<2x10x1024x2x32xf16>
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d7)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d6, d7)>
// CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[ATTENTION]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @attention_expand_all_masked(%arg0: tensor<20x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16, %arg4: tensor<20x4096x1024xf16>) -> tensor<2x10x2048x2x2x32xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5]} ins(%arg0, %arg1, %arg2, %arg3, %arg4: tensor<20x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16, tensor<20x4096x1024xf16>) outs(%0 : tensor<20x4096x64xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<20x4096x64xf16>
  %expanded = tensor.expand_shape %1 [[0, 1], [2, 3], [4, 5]] output_shape [2, 10, 2048, 2, 2, 32] : tensor<20x4096x64xf16> into tensor<2x10x2048x2x2x32xf16>
  util.return %expanded : tensor<2x10x2048x2x2x32xf16>
}

//CHECK-LABEL: func public @attention_expand_all_masked(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<20x4096x1024xf16>)
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x10x2048x2x2x32xf16>
//  CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1], [2, 3], [4]] output_shape [2, 10, 2048, 2, 16] : tensor<20x4096x16xf16> into tensor<2x10x2048x2x16xf16>
//  CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1], [2], [3]] output_shape [2, 10, 1024, 16] : tensor<20x1024x16xf16> into tensor<2x10x1024x16xf16>
//  CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1], [2], [3, 4]] output_shape [2, 10, 1024, 2, 32] : tensor<20x1024x64xf16> into tensor<2x10x1024x2x32xf16>
//  CHECK-DAG:   %[[MASK:.+]] = tensor.expand_shape %[[ARG4]] {{\[\[}}0, 1], [2, 3], [4]] output_shape [2, 10, 2048, 2, 1024] : tensor<20x4096x1024xf16> into tensor<2x10x2048x2x1024xf16>
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d7)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d5)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d6, d7)>
// CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]], %[[MASK]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[ATTENTION]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @attention_dynamic(%arg0: tensor<?x?x?xf16>, %arg1: tensor<?x?x?xf16>, %arg2: tensor<?x?x?xf16>, %arg3: f16) -> tensor<2x?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf16>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf16>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf16>
  %d3 = tensor.dim %arg1, %c1 : tensor<?x?x?xf16>
  %d4 = tensor.dim %arg2, %c2 : tensor<?x?x?xf16>
  %0 = tensor.empty(%d0, %d1, %d4) : tensor<?x?x?xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f16) outs(%0 : tensor<?x?x?xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<?x?x?xf16>
  %split = arith.divsi %d0, %c2 : index
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape[2, %split, %d1, %d4]
      : tensor<?x?x?xf16> into tensor<2x?x?x?xf16>
  util.return %expanded : tensor<2x?x?x?xf16>
}

//CHECK-LABEL: func public @attention_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[D4:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//  CHECK-DAG:   %[[SPLIT0:.+]] = arith.divsi %[[D0]]
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[SPLIT0]], %[[D1]], %[[D4]]) : tensor<2x?x?x?xf16>
//  CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[SPLIT0]], %[[D1]], %[[D2]]]
//  CHECK-DAG:   %[[D5:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[D6:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[D7:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[SPLIT1:.+]] = arith.divsi %[[D5]], %[[C2]]
//  CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[SPLIT1]], %[[D6]], %[[D7]]]
//  CHECK-DAG:   %[[D8:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG:   %[[D9:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//  CHECK-DAG:   %[[SPLIT2:.+]] = arith.divsi %[[D8]], %[[C2]]
//  CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[SPLIT2]], %[[D9]], %[[D4]]]
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
// CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[ATTENTION]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @attention_dynamic_masked(%arg0: tensor<?x?x?xf16>, %arg1: tensor<?x?x?xf16>, %arg2: tensor<?x?x?xf16>, %arg3: f16, %arg4: tensor<?x?x?xf16>) -> tensor<2x?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf16>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf16>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf16>
  %d3 = tensor.dim %arg1, %c1 : tensor<?x?x?xf16>
  %d4 = tensor.dim %arg2, %c2 : tensor<?x?x?xf16>
  %0 = tensor.empty(%d0, %d1, %d4) : tensor<?x?x?xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5]} ins(%arg0, %arg1, %arg2, %arg3, %arg4 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f16, tensor<?x?x?xf16>) outs(%0 : tensor<?x?x?xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<?x?x?xf16>
  %split = arith.divsi %d0, %c2 : index
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape[2, %split, %d1, %d4]
      : tensor<?x?x?xf16> into tensor<2x?x?x?xf16>
  util.return %expanded : tensor<2x?x?x?xf16>
}

//CHECK-LABEL: func public @attention_dynamic_masked(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[D4:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//  CHECK-DAG:   %[[SPLIT0:.+]] = arith.divsi %[[D0]]
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[SPLIT0]], %[[D1]], %[[D4]]) : tensor<2x?x?x?xf16>
//  CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[SPLIT0]], %[[D1]], %[[D2]]]
//  CHECK-DAG:   %[[D5:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[D6:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[D7:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[SPLIT1:.+]] = arith.divsi %[[D5]], %[[C2]]
//  CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[SPLIT1]], %[[D6]], %[[D7]]]
//  CHECK-DAG:   %[[D8:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG:   %[[D9:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//  CHECK-DAG:   %[[SPLIT2:.+]] = arith.divsi %[[D8]], %[[C2]]
//  CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[SPLIT2]], %[[D9]], %[[D4]]]
//  CHECK-DAG:   %[[D10:.+]] = tensor.dim %[[ARG4]], %[[C0]]
//  CHECK-DAG:   %[[D11:.+]] = tensor.dim %[[ARG4]], %[[C1]]
//  CHECK-DAG:   %[[D12:.+]] = tensor.dim %[[ARG4]], %[[C2]]
//  CHECK-DAG:   %[[SPLIT3:.+]] = arith.divsi %[[D10]], %[[C2]]
//  CHECK-DAG:   %[[MASK:.+]] = tensor.expand_shape %[[ARG4]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[SPLIT3]], %[[D11]], %[[D12]]]
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
// CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]], %[[MASK]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[ATTENTION]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @sink_through_attention(%0 : tensor<4x32x64x128xf16>, %1 : tensor<4x32x64x128xf16>, %2 : tensor<4x32x64x128xf16>, %cst : f16) -> (tensor<128x64x128xf16>) {
  %13 = tensor.empty() : tensor<4x32x64x128xf16>
  %collapsed_12 = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
  %collapsed_13 = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
  %collapsed_14 = tensor.collapse_shape %2 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
  %17 = tensor.empty() : tensor<128x64x128xf16>
  %18 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%collapsed_12, %collapsed_13, %collapsed_14, %cst : tensor<128x64x128xf16>, tensor<128x64x128xf16>, tensor<128x64x128xf16>, f16) outs(%17 : tensor<128x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<128x64x128xf16>
  util.return %18 : tensor<128x64x128xf16>
}

// CHECK-LABEL: util.func public @sink_through_attention
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG3:.+]]: f16
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//       CHECK:   %[[RET:.+]] = tensor.collapse_shape %[[ATTENTION]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
//       CHECK:   util.return %[[RET]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @sink_through_attention_masked(%0 : tensor<4x32x64x128xf16>, %1 : tensor<4x32x64x128xf16>, %2 : tensor<4x32x64x128xf16>, %cst : f16, %3 : tensor<4x32x64x64xf16>) -> (tensor<128x64x128xf16>) {
  %13 = tensor.empty() : tensor<4x32x64x128xf16>
  %collapsed_12 = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
  %collapsed_13 = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
  %collapsed_14 = tensor.collapse_shape %2 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
  %collapsed_15 = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<4x32x64x64xf16> into tensor<128x64x64xf16>
  %17 = tensor.empty() : tensor<128x64x128xf16>
  %18 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5]} ins(%collapsed_12, %collapsed_13, %collapsed_14, %cst, %collapsed_15 : tensor<128x64x128xf16>, tensor<128x64x128xf16>, tensor<128x64x128xf16>, f16, tensor<128x64x64xf16>) outs(%17 : tensor<128x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<128x64x128xf16>
  util.return %18 : tensor<128x64x128xf16>
}

// CHECK-LABEL: util.func public @sink_through_attention_masked
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG3:.+]]: f16
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]:
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]] :
//       CHECK:   %[[RET:.+]] = tensor.collapse_shape %[[ATTENTION]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
//       CHECK:   util.return %[[RET]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @sink_single_collapse(%0 : tensor<4x32x64x128xf16>, %1 : tensor<128x64x128xf16>, %2 : tensor<128x64x128xf16>, %cst : f16) -> (tensor<128x64x128xf16>) {
  %13 = tensor.empty() : tensor<4x32x64x128xf16>
  %collapsed_12 = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
  %17 = tensor.empty() : tensor<128x64x128xf16>
  %18 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%collapsed_12, %1, %2, %cst : tensor<128x64x128xf16>, tensor<128x64x128xf16>, tensor<128x64x128xf16>, f16) outs(%17 : tensor<128x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<128x64x128xf16>
  util.return %18 : tensor<128x64x128xf16>
}

// CHECK-LABEL: util.func public @sink_single_collapse
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG3:.+]]: f16
//   CHECK-DAG:   %[[EXPANDED1:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [4, 32, 64, 128]
//   CHECK-DAG:   %[[EXPANDED2:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [4, 32, 64, 128]
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:       ins(%[[ARG0]], %[[EXPANDED1]], %[[EXPANDED2]], %[[ARG3]] :
//       CHECK:   %[[RET:.+]] = tensor.collapse_shape %[[ATTENTION]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
//       CHECK:   util.return %[[RET]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @sink_single_collapse_masked(%0 : tensor<4x32x64x128xf16>, %1 : tensor<128x64x128xf16>, %2 : tensor<128x64x128xf16>, %cst : f16, %3 : tensor<128x64x64xf16>) -> (tensor<128x64x128xf16>) {
  %13 = tensor.empty() : tensor<4x32x64x128xf16>
  %collapsed_12 = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
  %17 = tensor.empty() : tensor<128x64x128xf16>
  %18 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5]} ins(%collapsed_12, %1, %2, %cst, %3 : tensor<128x64x128xf16>, tensor<128x64x128xf16>, tensor<128x64x128xf16>, f16, tensor<128x64x64xf16>) outs(%17 : tensor<128x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<128x64x128xf16>
  util.return %18 : tensor<128x64x128xf16>
}

// CHECK-LABEL: util.func public @sink_single_collapse_masked
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG3:.+]]: f16
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[EXPANDED1:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [4, 32, 64, 128]
//   CHECK-DAG:   %[[EXPANDED2:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [4, 32, 64, 128]
//   CHECK-DAG:   %[[EXPANDED3:.+]] = tensor.expand_shape %[[ARG4]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [4, 32, 64, 64]
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:       ins(%[[ARG0]], %[[EXPANDED1]], %[[EXPANDED2]], %[[ARG3]], %[[EXPANDED3]] :
//       CHECK:   %[[RET:.+]] = tensor.collapse_shape %[[ATTENTION]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
//       CHECK:   util.return %[[RET]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

util.func public @sink_through_k2(%0 : tensor<128x16x2x2x128xf16>, %1 : tensor<128x64x128xf16>, %2 : tensor<128x64x128xf16>, %cst : f16) -> (tensor<128x64x128xf16>) {
  %13 = tensor.empty() : tensor<4x32x64x128xf16>
  %collapsed_12 = tensor.collapse_shape %0 [[0], [1, 2, 3], [4]] : tensor<128x16x2x2x128xf16> into tensor<128x64x128xf16>
  %17 = tensor.empty() : tensor<128x64x128xf16>
  %18 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%2, %1, %collapsed_12, %cst : tensor<128x64x128xf16>, tensor<128x64x128xf16>, tensor<128x64x128xf16>, f16) outs(%17 : tensor<128x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<128x64x128xf16>
  util.return %18 : tensor<128x64x128xf16>
}

// CHECK-LABEL: util.func public @sink_through_k2
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG3:.+]]: f16
//   CHECK-DAG:   %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG1]]
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d3, d4, d5, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d3, d4, d5, d6)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d6)>
//  CHECK-SAME:       ins(%[[ARG2]], %[[EXPANDED]], %[[ARG0]], %[[ARG3]] :
//       CHECK:   util.return %[[ATTENTION]]

// -----

util.func @scatter_collapse_updates(%arg0: tensor<4x?x2x16x4x128xf16>, %arg1: tensor<?x1xi32>, %arg2: tensor<?x2x16x4x128xf16>) -> tensor<?x2x16x4x128xf16> {
  %collapsed = tensor.collapse_shape %arg0[[0, 1], [2], [3], [4], [5]] : tensor<4x?x2x16x4x128xf16> into tensor<?x2x16x4x128xf16>
  %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%collapsed, %arg1 : tensor<?x2x16x4x128xf16>, tensor<?x1xi32>) outs(%arg2 : tensor<?x2x16x4x128xf16>) {
  ^bb0(%arg7: f16, %arg8: f16):
    iree_linalg_ext.yield %arg7 : f16
  } -> tensor<?x2x16x4x128xf16>
  util.return %1 : tensor<?x2x16x4x128xf16>
}

// CHECK-LABEL: util.func public @scatter_collapse_updates
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//       CHECK:   %[[INDICES:.+]] = tensor.expand_shape
//  CHECK-SAME:     tensor<?x1xi32> into tensor<4x?x1xi32>
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[ARG0]], %[[INDICES]]
//  CHECK-SAME:       outs(%[[ARG2]]
//       CHECK:   util.return %[[SCATTER]]

// -----

util.func @scatter_collapse_updates_partial(%arg0: tensor<4x?x2x2x16x4x128xf16>, %arg1: tensor<?x1xi32>, %arg2: tensor<10x16x4x128xf16>) -> tensor<10x16x4x128xf16> {
  %collapsed = tensor.collapse_shape %arg0[[0, 1, 2, 3], [4], [5], [6]] : tensor<4x?x2x2x16x4x128xf16> into tensor<?x16x4x128xf16>
  %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%collapsed, %arg1 : tensor<?x16x4x128xf16>, tensor<?x1xi32>) outs(%arg2 : tensor<10x16x4x128xf16>) {
  ^bb0(%arg7: f16, %arg8: f16):
    iree_linalg_ext.yield %arg7 : f16
  } -> tensor<10x16x4x128xf16>
  util.return %1 : tensor<10x16x4x128xf16>
}

// CHECK-LABEL: util.func public @scatter_collapse_updates_partial
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[INDICES:.+]] = tensor.expand_shape %[[ARG1]] {{.*}} tensor<?x1xi32> into tensor<4x?x2x2x1xi32>
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[ARG0]], %[[INDICES]]
//  CHECK-SAME:       outs(%[[ARG2]]
//       CHECK:   util.return %[[SCATTER]]

// -----

util.func @scatter_collapse_original(%arg0: tensor<?x1x32x8x128xf16>, %arg1: tensor<?x1xi32>, %arg2: tensor<?x2x16x4x2x64x2xf16>, %arg3 : index) -> tensor<?x32x8x128xf16> {
    %collapsed = tensor.collapse_shape %arg2 [[0], [1, 2], [3, 4], [5, 6]] : tensor<?x2x16x4x2x64x2xf16> into tensor<?x32x8x128xf16>
    %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%arg0, %arg1 : tensor<?x1x32x8x128xf16>, tensor<?x1xi32>) outs(%collapsed : tensor<?x32x8x128xf16>) {
    ^bb0(%arg6: f16, %arg7: f16):
      iree_linalg_ext.yield %arg6 : f16
    } -> tensor<?x32x8x128xf16>
  util.return %1 : tensor<?x32x8x128xf16>
}

// CHECK-LABEL: util.func public @scatter_collapse_original
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//       CHECK:   %[[UPDATES:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<?x1x32x8x128xf16> into tensor<?x1x2x16x4x2x64x2xf16>
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[UPDATES]], %[[ARG1]]
//  CHECK-SAME:       outs(%[[ARG2]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[SCATTER]]
//       CHECK:   util.return %[[COLLAPSE]]

// -----

util.func @scatter_original_noop(%arg0: tensor<?x1x32x8x128xf16>, %arg1: tensor<?x1xi32>, %arg2: tensor<?x80x2x32x8x128xf16>, %arg3 : index) -> tensor<?x32x8x128xf16> {
    %collapsed = tensor.collapse_shape %arg2 [[0, 1, 2], [3], [4], [5]] : tensor<?x80x2x32x8x128xf16> into tensor<?x32x8x128xf16>
    %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%arg0, %arg1 : tensor<?x1x32x8x128xf16>, tensor<?x1xi32>) outs(%collapsed : tensor<?x32x8x128xf16>) {
    ^bb0(%arg6: f16, %arg7: f16):
      iree_linalg_ext.yield %arg6 : f16
    } -> tensor<?x32x8x128xf16>
  util.return %1 : tensor<?x32x8x128xf16>
}

// CHECK-LABEL: util.func public @scatter_original_noop
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG2]]
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]]
//  CHECK-SAME:       outs(%[[COLLAPSE]]
//       CHECK:   util.return %[[SCATTER]]


// -----

util.func @scatter_collapse_original_partial(%arg0: tensor<?x1x32x8x128xf16>, %arg1: tensor<?x1xi32>, %arg2: tensor<5x?x2x16x4x2x64x2xf16>, %arg3 : index) -> tensor<?x32x8x128xf16> {
    %collapsed = tensor.collapse_shape %arg2 [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<5x?x2x16x4x2x64x2xf16> into tensor<?x32x8x128xf16>
    %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%arg0, %arg1 : tensor<?x1x32x8x128xf16>, tensor<?x1xi32>) outs(%collapsed : tensor<?x32x8x128xf16>) {
    ^bb0(%arg6: f16, %arg7: f16):
      iree_linalg_ext.yield %arg6 : f16
    } -> tensor<?x32x8x128xf16>
  util.return %1 : tensor<?x32x8x128xf16>
}

// CHECK-LABEL: util.func public @scatter_collapse_original_partial
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[UPDATES:.+]] = tensor.expand_shape %[[ARG0]] {{.*}} tensor<?x1x32x8x128xf16> into tensor<?x1x2x16x4x2x64x2xf16>
//   CHECK-DAG:   %[[ORIGINAL:.+]] = tensor.collapse_shape %[[ARG2]] {{.*}} tensor<5x?x2x16x4x2x64x2xf16> into tensor<?x2x16x4x2x64x2xf16>
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[UPDATES]], %[[ARG1]]
//  CHECK-SAME:       outs(%[[ORIGINAL]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[SCATTER]]
//       CHECK:   util.return %[[COLLAPSE]]

// -----

util.func @scatter_collapse_indices(%arg0: tensor<?x2x16x4x128xf16>, %arg1: tensor<4x?x1xi32>, %arg2: tensor<?x2x16x4x128xf16>) -> tensor<?x2x16x4x128xf16> {
  %collapsed = tensor.collapse_shape %arg1[[0, 1], [2]] : tensor<4x?x1xi32> into tensor<?x1xi32>
  %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%arg0, %collapsed: tensor<?x2x16x4x128xf16>, tensor<?x1xi32>) outs(%arg2 : tensor<?x2x16x4x128xf16>) {
  ^bb0(%arg7: f16, %arg8: f16):
    iree_linalg_ext.yield %arg7 : f16
  } -> tensor<?x2x16x4x128xf16>
  util.return %1 : tensor<?x2x16x4x128xf16>
}

// CHECK-LABEL: util.func public @scatter_collapse_indices
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[UPDATES:.+]] = tensor.expand_shape %[[ARG0]] {{.*}} tensor<?x2x16x4x128xf16> into tensor<4x?x2x16x4x128xf16>
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[UPDATES]], %[[ARG1]]
//  CHECK-SAME:       outs(%[[ARG2]]
//       CHECK:   util.return %[[SCATTER]]

// -----

util.func @scatter_collapse_indices_partial(%arg0: tensor<?x2x16x4x128xf16>, %arg1: tensor<4x?x1x1xi32>, %arg2: tensor<?x2x16x4x128xf16>) -> tensor<?x2x16x4x128xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x?x1x1xi32>) outs(%arg1 : tensor<4x?x1x1xi32>){
^bb0(%in : i32, %out : i32):
    linalg.yield %in : i32
  } -> tensor<4x?x1x1xi32>
  %collapsed = tensor.collapse_shape %0[[0, 1], [2, 3]] : tensor<4x?x1x1xi32> into tensor<?x1xi32>
  %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%arg0, %collapsed: tensor<?x2x16x4x128xf16>, tensor<?x1xi32>) outs(%arg2 : tensor<?x2x16x4x128xf16>) {
  ^bb0(%arg7: f16, %arg8: f16):
    iree_linalg_ext.yield %arg7 : f16
  } -> tensor<?x2x16x4x128xf16>
  util.return %1 : tensor<?x2x16x4x128xf16>
}

// CHECK-LABEL: util.func public @scatter_collapse_indices_partial
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[UPDATES:.+]] = tensor.expand_shape {{.*}} tensor<?x2x16x4x128xf16> into tensor<4x?x2x16x4x128xf16>
//   CHECK-DAG:   %[[ORIGINAL:.+]] = tensor.collapse_shape {{.*}} tensor<4x?x1x1xi32> into tensor<4x?x1xi32>
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[UPDATES]], %[[ORIGINAL]]
//  CHECK-SAME:       outs(%[[ARG2]]
//       CHECK:   util.return %[[SCATTER]]

// -----

util.func public @scatter_collapse(%arg0: tensor<?x2x16x4x128xf16>, %arg1: tensor<?x1xi32>, %arg2: tensor<?x2x16x4x128xf16>) -> tensor<?x2x16x4x4x32xf16> {
  %c0 = arith.constant 0 : index
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%arg0, %arg1 : tensor<?x2x16x4x128xf16>, tensor<?x1xi32>) outs(%arg2 : tensor<?x2x16x4x128xf16>) {
  ^bb0(%arg3: f16, %arg4: f16):
    iree_linalg_ext.yield %arg3 : f16
  } -> tensor<?x2x16x4x128xf16>
  %dim = tensor.dim %arg0, %c0 : tensor<?x2x16x4x128xf16>
  %expanded = tensor.expand_shape %0 [[0], [1], [2], [3], [4, 5]] output_shape [%dim, 2, 16, 4, 4, 32] : tensor<?x2x16x4x128xf16> into tensor<?x2x16x4x4x32xf16>
  util.return %expanded : tensor<?x2x16x4x4x32xf16>
}
// CHECK-LABEL: util.func public @scatter_collapse
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[UPDATES:.+]] = tensor.expand_shape %[[ARG0]] {{.*}} tensor<?x2x16x4x128xf16> into tensor<?x2x16x4x4x32xf16>
//   CHECK-DAG:   %[[ORIGINAL:.+]] = tensor.expand_shape %[[ARG2]] {{.*}} tensor<?x2x16x4x128xf16> into tensor<?x2x16x4x4x32xf16>
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[UPDATES]], %[[ARG1]]
//  CHECK-SAME:       outs(%[[ORIGINAL]]
//       CHECK:   util.return %[[SCATTER]]

// -----

util.func public @scatter_collapse_noop(%arg0: tensor<10xf16>, %arg1: tensor<10x1xi32>, %arg2: tensor<128xf16>) -> tensor<4x4x4x2xf16> {
  %c0 = arith.constant 0 : index
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%arg0, %arg1 : tensor<10xf16>, tensor<10x1xi32>) outs(%arg2 : tensor<128xf16>) {
  ^bb0(%arg3: f16, %arg4: f16):
    iree_linalg_ext.yield %arg3 : f16
  } -> tensor<128xf16>
  %expanded = tensor.expand_shape %0 [[0, 1, 2, 3]] output_shape[4, 4, 4, 2] : tensor<128xf16> into tensor<4x4x4x2xf16>
  util.return %expanded : tensor<4x4x4x2xf16>
}
// CHECK-LABEL: util.func public @scatter_collapse_noop
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]]
//  CHECK-SAME:       outs(%[[ARG2]]
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[SCATTER]]
//       CHECK:   util.return %[[EXPANDED]]

// -----

util.func public @gather_expand(%arg0: tensor<100x128xf16>, %arg1: tensor<10xi32>, %arg2: tensor<10x128xf16>) -> tensor<2x5x4x32xf16> {
  %c0 = arith.constant 0 : index
  %0 = iree_linalg_ext.gather dimension_map = [0] ins(%arg0, %arg1 : tensor<100x128xf16>, tensor<10xi32>) outs(%arg2 : tensor<10x128xf16>) -> tensor<10x128xf16>
  %expanded = tensor.expand_shape %0[[0, 1], [2, 3]] output_shape[2, 5, 4, 32] : tensor<10x128xf16> into tensor<2x5x4x32xf16>
  util.return %expanded : tensor<2x5x4x32xf16>
}
// CHECK-LABEL: util.func public @gather_expand
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[EXPANDED0:.+]] = tensor.expand_shape %[[ARG0]] {{.+}} tensor<100x128xf16> into tensor<100x4x32xf16>
//   CHECK-DAG:   %[[EXPANDED1:.+]] = tensor.expand_shape %[[ARG2]] {{.+}} tensor<10x128xf16> into tensor<2x5x4x32xf16>
//   CHECK-DAG:   %[[EXPANDED2:.+]] = tensor.expand_shape %[[ARG1]] {{.+}} tensor<10xi32> into tensor<2x5xi32>
//       CHECK:   %[[GATHER:.+]] = iree_linalg_ext.gather
//  CHECK-SAME:       ins(%[[EXPANDED0]], %[[EXPANDED2]]
//  CHECK-SAME:       outs(%[[EXPANDED1]]
//       CHECK:   util.return %[[GATHER]]

// -----

util.func public @gather_expand_partial(%arg0: tensor<100x128xf16>, %arg1: tensor<10xi32>, %arg2: tensor<10x128xf16>) -> tensor<10x4x32xf16> {
  %c0 = arith.constant 0 : index
  %0 = iree_linalg_ext.gather dimension_map = [0] ins(%arg0, %arg1 : tensor<100x128xf16>, tensor<10xi32>) outs(%arg2 : tensor<10x128xf16>) -> tensor<10x128xf16>
  %expanded = tensor.expand_shape %0[[0], [1, 2]] output_shape[10, 4, 32] : tensor<10x128xf16> into tensor<10x4x32xf16>
  util.return %expanded : tensor<10x4x32xf16>
}
// CHECK-LABEL: util.func public @gather_expand
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[EXPANDED0:.+]] = tensor.expand_shape %[[ARG0]] {{.+}} tensor<100x128xf16> into tensor<100x4x32xf16>
//   CHECK-DAG:   %[[EXPANDED1:.+]] = tensor.expand_shape %[[ARG2]] {{.+}} tensor<10x128xf16> into tensor<10x4x32xf16>
//       CHECK:   %[[GATHER:.+]] = iree_linalg_ext.gather
//  CHECK-SAME:       ins(%[[EXPANDED0]], %[[ARG1]]
//  CHECK-SAME:       outs(%[[EXPANDED1]]
//       CHECK:   util.return %[[GATHER]]

// -----

util.func public @gather_collapse_source(%arg0: tensor<10x10x4x32xf16>, %arg1: tensor<10xi32>, %arg2: tensor<10x128xf16>) -> tensor<10x128xf16> {
  %c0 = arith.constant 0 : index
  %collapse = tensor.collapse_shape %arg0[[0, 1], [2, 3]] : tensor<10x10x4x32xf16> into tensor<100x128xf16>
  %0 = iree_linalg_ext.gather dimension_map = [0] ins(%collapse, %arg1 : tensor<100x128xf16>, tensor<10xi32>) outs(%arg2 : tensor<10x128xf16>) -> tensor<10x128xf16>
  util.return %0 : tensor<10x128xf16>
}
// CHECK-LABEL: util.func public @gather_collapse_source
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[EXPANDED0:.+]] = tensor.collapse_shape %[[ARG0]] {{.+}} tensor<10x10x4x32xf16> into tensor<100x4x32xf16>
//   CHECK-DAG:   %[[EXPANDED1:.+]] = tensor.expand_shape %[[ARG2]] {{.+}} tensor<10x128xf16> into tensor<10x4x32xf16>
//       CHECK:   %[[GATHER:.+]] = iree_linalg_ext.gather
//  CHECK-SAME:       ins(%[[EXPANDED0]], %[[ARG1]]
//  CHECK-SAME:       outs(%[[EXPANDED1]]
//       CHECK:   tensor.collapse_shape %[[GATHER]] {{.*}} tensor<10x4x32xf16> into tensor<10x128xf16>
