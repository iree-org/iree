// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-preprocessing-fold-attention-with-transpose, resolve-shaped-type-result-dims))" --split-input-file --mlir-print-local-scope  %s | FileCheck %s

util.func public @fuse_attention_expand_transpose(
  %arg0: tensor<?x?x?xf16>, %arg1 : tensor<?x?x?xf16>, %arg2 : tensor<?x?x?xf16>, %arg3 : f16) -> tensor<2x?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf16>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf16>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf16>
  %d3 = tensor.dim %arg1, %c1 : tensor<?x?x?xf16>
  %d4 = tensor.dim %arg2, %c2 : tensor<?x?x?xf16>
  %empty = tensor.empty(%d0, %d1, %d4) : tensor<?x?x?xf16>
  %attention = iree_linalg_ext.attention {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
    ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f16)
    outs(%empty : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %split = arith.divsi %d0, %c2 : index
  %expanded = tensor.expand_shape %attention [[0, 1], [2], [3]] output_shape[2, %split, %d1, %d4]
      : tensor<?x?x?xf16> into tensor<2x?x?x?xf16>
  %empty2 = tensor.empty(%d1, %split, %d4) : tensor<2x?x?x?xf16>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<2x?x?x?xf16>) outs(%empty2 : tensor<2x?x?x?xf16>) {
    ^bb0(%b0 : f16, %b1 : f16):
      linalg.yield %b0 : f16
  } -> tensor<2x?x?x?xf16>
  util.return %transpose : tensor<2x?x?x?xf16>
}
// CHECK-LABEL: func public @fuse_attention_expand_transpose(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
//  CHECK-SAME:     %[[ARG3:.+]]: f16)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[D4:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//   CHECK-DAG:   %[[D_SPLIT:.+]] = arith.divsi %[[D0]], %[[C2]]
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[D1]], %[[D_SPLIT]], %[[D4]]) : tensor<2x?x?x?xf16>
//   CHECK-DAG:   %[[D_SPLIT2:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%[[D0]]]
//   CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//   CHECK-DAG:   %[[D3:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[D_SPLIT2]], %[[D1]], %[[D2]]{{\]}}
//   CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[D_SPLIT2]], %[[D3]], %[[D2]]{{\]}}
//   CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[D_SPLIT2]], %[[D3]], %[[D4]]{{\]}}
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:           [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d5)>]
//  CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[EMPTY]] :
//       CHECK:   util.return %[[ATTENTION]]

// -----

util.func public @fuse_attention_expand_transpose_static(
      %arg0 : tensor<20x4096x16xf16>, %arg1 : tensor<20x1024x16xf16>,
      %arg2 : tensor<20x1024x64xf16>, %arg3 : f16) -> tensor<2x4096x10x64xf16> {
  %empty = tensor.empty() : tensor<20x4096x64xf16>
  %attention = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
      ins(%arg0, %arg1, %arg2, %arg3 : tensor<20x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16)
      outs(%empty: tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16>
  %expanded = tensor.expand_shape %attention [[0, 1], [2], [3]]
      output_shape [2, 10, 4096, 64] : tensor<20x4096x64xf16> into tensor<2x10x4096x64xf16>
  %empty2 = tensor.empty() : tensor<2x4096x10x64xf16>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<2x10x4096x64xf16>) outs(%empty2 : tensor<2x4096x10x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
  } -> tensor<2x4096x10x64xf16>
  util.return %transpose : tensor<2x4096x10x64xf16>
}
// CHECK-LABEL: func public @fuse_attention_expand_transpose_static(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x4096x16xf16>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
//  CHECK-SAME:     %[[ARG3:.+]]: f16)
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x4096x10x64xf16>
//   CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 4096, 16]
//   CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 1024, 16]
//   CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 1024, 64]
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:           [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d5)>]
//  CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[EMPTY]] :
//       CHECK:   util.return %[[ATTENTION]]
