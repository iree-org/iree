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
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
    ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f16)
    outs(%empty : tensor<?x?x?xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<?x?x?xf16>
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
//   CHECK-DAG:   %[[D:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[D]], %[[D0]], %[[D1]]) : tensor<?x?x?xf16>
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:           [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4) -> ()>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[EMPTY]] :
//   CHECK-DAG:   %[[D_SPLIT:.+]] = arith.divsi %[[D]], %[[C2]]
//   CHECK-DAG:   %[[EXPANDED:.+]] = tensor.expand_shape %[[ATTENTION]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[D_SPLIT]], %[[D0]], %[[D1]]]
//   CHECK-DAG:   %[[OUTS:.+]] = tensor.empty(%[[D0]], %[[D_SPLIT]], %[[D1]]) : tensor<2x?x?x?xf16>
//   CHECK-DAG:   %[[TRANSPOSE:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:           [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>]
//  CHECK-SAME:       ins(%[[EXPANDED]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   linalg.yield
//       CHECK:   util.return %[[TRANSPOSE]]

// -----

util.func public @fuse_attention_expand_transpose_static(
      %arg0 : tensor<20x4096x16xf16>, %arg1 : tensor<20x1024x16xf16>,
      %arg2 : tensor<20x1024x64xf16>, %arg3 : f16) -> tensor<2x4096x10x64xf16> {
  %empty = tensor.empty() : tensor<20x4096x64xf16>
  %attention = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
      ins(%arg0, %arg1, %arg2, %arg3 : tensor<20x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16)
      outs(%empty: tensor<20x4096x64xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<20x4096x64xf16>
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
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<20x4096x64xf16>
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:           [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4) -> ()>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[EMPTY]] :
//   CHECK-DAG:   %[[EXPANDED:.+]] = tensor.expand_shape %[[ATTENTION]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 4096, 64]
//   CHECK-DAG:   %[[OUTS:.+]] = tensor.empty() : tensor<2x4096x10x64xf16>
//   CHECK-DAG:   %[[TRANSPOSE:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps =
//  CHECK-SAME:           [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
//  CHECK-SAME:            affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>]
//  CHECK-SAME:       ins(%[[EXPANDED]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   linalg.yield
//       CHECK:   util.return %[[TRANSPOSE]]
