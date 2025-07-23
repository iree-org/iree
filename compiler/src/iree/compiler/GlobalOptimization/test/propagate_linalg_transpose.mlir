// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose{enable-aggressive-propagation=true}))" --split-input-file %s | FileCheck %s --check-prefix=APROP
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose{test-sinking-only=true}))" --split-input-file %s | FileCheck %s --check-prefix=SINK
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose{test-bubbling-only=true}))" --split-input-file %s | FileCheck %s --check-prefix=BUBBLE
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose{enable-aggressive-propagation-through-conv=true}))" --split-input-file %s | FileCheck %s --check-prefix=CONV

util.func public @specialize_transpose_op(%arg0 : tensor<1x2x3xf32>,
                                   %empty : tensor<3x2x1xf32>) -> tensor<3x2x1xf32> {
  %transposed = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                    affine_map<(d0, d1, d2) -> (d1, d2, d0)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%arg0 : tensor<1x2x3xf32>)
                outs(%empty : tensor<3x2x1xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    linalg.yield %in : f32
                  } -> tensor<3x2x1xf32>
  util.return %transposed : tensor<3x2x1xf32>
}
// CHECK-LABEL: util.func public @specialize_transpose_op
//       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose
//  CHECK-SAME:     permutation = [2, 1, 0]
//       CHECK:   util.return %[[TRANSPOSE]]

// -----

util.func public @specialize_non_involution_transpose_op(%arg0 : tensor<1x2x3xf32>,
                                   %empty : tensor<2x3x1xf32>) -> tensor<2x3x1xf32> {
  %transposed = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d2, d0, d1)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%arg0 : tensor<1x2x3xf32>)
                outs(%empty : tensor<2x3x1xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    linalg.yield %in : f32
                  } -> tensor<2x3x1xf32>
  util.return %transposed : tensor<2x3x1xf32>
}
// CHECK-LABEL: util.func public @specialize_non_involution_transpose_op
//       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose
//  CHECK-SAME:     permutation = [1, 2, 0]
//       CHECK:   util.return %[[TRANSPOSE]]

// -----

util.func public @fold_transpose_of_fill() -> tensor<32x128xf32> {
  %cst = arith.constant 1.0 : f32
  %empty = tensor.empty(): tensor<128x32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x32xf32>) -> tensor<128x32xf32>
  %empty_t = tensor.empty(): tensor<32x128xf32>
  %transposed = linalg.transpose ins(%fill : tensor<128x32xf32>)
      outs(%empty_t : tensor<32x128xf32>) permutation = [1, 0]
  util.return %transposed : tensor<32x128xf32>
}
// CHECK-LABEL: util.func public @fold_transpose_of_fill
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   util.return %[[FILL]]

// -----

util.func public @propagate_through_extract_slice(%arg0 : tensor<1x256x128xf32>) -> tensor<1x128x32xf32> {
  %empty = tensor.empty(): tensor<1x128x256xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<1x256x128xf32>)
      outs(%empty : tensor<1x128x256xf32>) permutation = [0, 2, 1]
  %slice = tensor.extract_slice %transposed[0, 0, 0] [1, 128, 32] [1, 1, 1] : tensor<1x128x256xf32> to tensor<1x128x32xf32>
  util.return %slice : tensor<1x128x32xf32>
}
// CHECK-LABEL: util.func public @propagate_through_extract_slice
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice {{.*}}[0, 0, 0] [1, 32, 128] [1, 1, 1]
//  CHECK-SAME:                     tensor<1x256x128xf32> to tensor<1x32x128xf32>
//       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[SLICE]] : tensor<1x32x128xf32>)
//  CHECK-SAME:     permutation = [0, 2, 1]
//       CHECK:   util.return %[[TRANSPOSE]]

// -----

util.func public @propagate_through_rank_reduced_extract_slice(%arg0 : tensor<1x256x1x128x1xf32>) -> tensor<128x32xf32> {
  %empty = tensor.empty(): tensor<1x128x1x256x1xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<1x256x1x128x1xf32>)
      outs(%empty : tensor<1x128x1x256x1xf32>) permutation = [0, 3, 2, 1, 4]
  %slice = tensor.extract_slice %transposed[0, 0, 0, 0, 0] [1, 128, 1, 32, 1] [1, 1, 1, 1, 1]
             : tensor<1x128x1x256x1xf32> to tensor<128x32xf32>
  util.return %slice : tensor<128x32xf32>
}
// CHECK-LABEL: util.func public @propagate_through_rank_reduced_extract_slice
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice
//  CHECK-SAME:                     [0, 0, 0, 0, 0] [1, 32, 1, 128, 1] [1, 1, 1, 1, 1]
//  CHECK-SAME:                     tensor<1x256x1x128x1xf32> to tensor<32x128xf32>
//       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[SLICE]] : tensor<32x128xf32>)
//  CHECK-SAME:     permutation = [1, 0]
//       CHECK:   util.return %[[TRANSPOSE]]

// -----

util.func public @rank_reduced_extract_transposed_unit_dim(%arg0: tensor<256x1x32x128xf32>, %arg1: tensor<1x32x256x128xf32>) -> tensor<32x64x128xf32> {
  %transposed = linalg.transpose ins(%arg0 : tensor<256x1x32x128xf32>) outs(%arg1 : tensor<1x32x256x128xf32>) permutation = [1, 2, 0, 3]
  %extracted_slice = tensor.extract_slice %transposed[0, 0, 0, 0] [1, 32, 64, 128] [1, 1, 1, 1] : tensor<1x32x256x128xf32> to tensor<32x64x128xf32>
  util.return %extracted_slice : tensor<32x64x128xf32>
}
// SINK-LABEL: util.func public @rank_reduced_extract_transposed_unit_dim
//       SINK:   %[[EXT:.+]] = tensor.extract_slice
//  SINK-SAME:                   tensor<256x1x32x128xf32> to tensor<64x32x128xf32>
//       SINK:   %[[RES:.+]] = linalg.transpose ins(%[[EXT]] : tensor<64x32x128xf32>
//  SINK-SAME:                    outs({{.*}} : tensor<32x64x128xf32>)
//  SINK-SAME:                    permutation = [1, 0, 2]
//       SINK:   util.return %[[RES]] : tensor<32x64x128xf32>

// -----

util.func public @propagate_to_matmul_ops(%lhs: tensor<16x16xf32>,
                                   %transposed_a: tensor<16x16xf32>,
                                   %transposed_b: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %empty = tensor.empty(): tensor<16x16xf32>
  %rhs = linalg.transpose ins(%transposed_b : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  %first_mm = linalg.matmul ins(%lhs, %rhs : tensor<16x16xf32>, tensor<16x16xf32>)
                            outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>

  %second_lhs = linalg.transpose ins(%transposed_a : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  %second_mm = linalg.matmul ins(%second_lhs, %first_mm : tensor<16x16xf32>, tensor<16x16xf32>)
                            outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  util.return %second_mm : tensor<16x16xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//   CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
//   CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-LABEL: util.func public @propagate_to_matmul_ops
//       CHECK:   linalg.matmul
//  CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//       CHECK:   %[[SECOND_MM:.+]] = linalg.matmul
//  CHECK-SAME:     indexing_maps = [#[[$MAP3]], #[[$MAP4]], #[[$MAP2]]]
//       CHECK:   util.return %[[SECOND_MM]]

// -----

util.func public @propagate_to_transposed_matmul_ops(%lhs: tensor<16x16xf32>,
                                              %second_lhs: tensor<16x16xf32>,
                                              %rhs: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %empty = tensor.empty(): tensor<16x16xf32>
  %transpose_b = linalg.transpose ins(%rhs : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  %first_mm = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%lhs, %transpose_b : tensor<16x16xf32>, tensor<16x16xf32>)
    outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>

  %transpose_a = linalg.transpose ins(%second_lhs : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  %second_mm = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d2, d0)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%transpose_a, %first_mm : tensor<16x16xf32>, tensor<16x16xf32>)
    outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  util.return %second_mm : tensor<16x16xf32>
}
// CHECK-LABEL: util.func public @propagate_to_transposed_matmul_ops
//       CHECK:   linalg.matmul ins
//       CHECK:   %[[SECOND_MM:.+]] = linalg.matmul ins
//       CHECK:   util.return %[[SECOND_MM]]

// -----

util.func public @propagate_to_bmm_ops(%lhs: tensor<2x16x16xf32>,
                                   %transposed_a: tensor<2x16x16xf32>,
                                   %transposed_b: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %empty = tensor.empty(): tensor<2x16x16xf32>
  %rhs = linalg.transpose ins(%transposed_b : tensor<2x16x16xf32>)
      outs(%empty : tensor<2x16x16xf32>) permutation = [0, 2, 1]
  %first_bmm = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x16x16xf32>, tensor<2x16x16xf32>)
                            outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

  %second_lhs = linalg.transpose ins(%transposed_a : tensor<2x16x16xf32>)
      outs(%empty : tensor<2x16x16xf32>) permutation = [0, 2, 1]
  %second_bmm = linalg.batch_matmul ins(%second_lhs, %first_bmm : tensor<2x16x16xf32>, tensor<2x16x16xf32>)
                            outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  util.return %second_bmm : tensor<2x16x16xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//   CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
//   CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-LABEL: util.func public @propagate_to_bmm_ops
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//       CHECK:   %[[SECOND_MM:.+]] = linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [#[[$MAP3]], #[[$MAP4]], #[[$MAP2]]]
//       CHECK:   util.return %[[SECOND_MM]]

// -----

util.func public @propagate_to_transposed_bmm_ops(%lhs: tensor<2x16x16xf32>,
                                              %second_lhs: tensor<2x16x16xf32>,
                                              %rhs: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %empty = tensor.empty(): tensor<2x16x16xf32>
  %transpose_b = linalg.transpose ins(%rhs : tensor<2x16x16xf32>)
      outs(%empty : tensor<2x16x16xf32>) permutation = [0, 2, 1]
  %first_bmm = linalg.batch_matmul
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ]
    ins(%lhs, %transpose_b : tensor<2x16x16xf32>, tensor<2x16x16xf32>)
    outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

  %transpose_a = linalg.transpose ins(%second_lhs : tensor<2x16x16xf32>)
      outs(%empty : tensor<2x16x16xf32>) permutation = [0, 2, 1]
  %second_bmm = linalg.batch_matmul
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ]
    ins(%transpose_a, %first_bmm : tensor<2x16x16xf32>, tensor<2x16x16xf32>)
    outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  util.return %second_bmm : tensor<2x16x16xf32>
}
// CHECK-LABEL: util.func public @propagate_to_transposed_bmm_ops
//       CHECK:   linalg.batch_matmul ins
//       CHECK:   %[[SECOND_MM:.+]] = linalg.batch_matmul ins
//       CHECK:   util.return %[[SECOND_MM]]

// -----

util.func public @do_not_propagate_to_matmul_in_dispatch(%lhs: tensor<16x16xf32>,
                                                  %transposed_b: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %empty = tensor.empty(): tensor<16x16xf32>
  %rhs = linalg.transpose ins(%transposed_b : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  %dispatch = flow.dispatch.region[] -> (tensor<16x16xf32>) {
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<16x16xf32>, tensor<16x16xf32>)
                              outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    flow.return %mm : tensor<16x16xf32>
  }
  util.return %dispatch : tensor<16x16xf32>
}
// CHECK-LABEL: util.func public @do_not_propagate_to_matmul_in_dispatch
//       CHECK:   linalg.transpose
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     linalg.matmul ins
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func public @propagate_to_bmm_transpose_batch(%transposed_lhs: tensor<16x2x16xf32>,
                                            %rhs: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %empty = tensor.empty(): tensor<2x16x16xf32>
  %lhs = linalg.transpose ins(%transposed_lhs : tensor<16x2x16xf32>)
      outs(%empty : tensor<2x16x16xf32>) permutation = [1, 0, 2]
  %bmm = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x16x16xf32>, tensor<2x16x16xf32>)
                            outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  util.return %bmm : tensor<2x16x16xf32>
}
// APROP: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// APROP: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// APROP: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// APROP-LABEL: util.func public @propagate_to_bmm_transpose_batch
//  APROP-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<16x2x16xf32>
//  APROP-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<2x16x16xf32>
//       APROP:   %[[GENERIC:.+]] = linalg.generic
//  APROP-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  APROP-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  APROP-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<16x2x16xf32>, tensor<2x16x16xf32>
//       APROP:   util.return %[[GENERIC]] : tensor<2x16x16xf32>

// -----

util.func public @do_not_propagate_to_conv(%transposed_lhs: tensor<18x2x18x8xf32>,
                                           %rhs: tensor<3x3x8x32xf32>) -> tensor<2x16x16x32xf32> {
  %empty = tensor.empty(): tensor<2x18x18x8xf32>
  %lhs = linalg.transpose ins(%transposed_lhs : tensor<18x2x18x8xf32>)
      outs(%empty : tensor<2x18x18x8xf32>) permutation = [1, 0, 2, 3]
  %out = tensor.empty(): tensor<2x16x16x32xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {strides = dense<1> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>}
    ins(%lhs, %rhs : tensor<2x18x18x8xf32>, tensor<3x3x8x32xf32>)
    outs(%out : tensor<2x16x16x32xf32>) -> tensor<2x16x16x32xf32>
  util.return %conv : tensor<2x16x16x32xf32>
}

// APROP-LABEL: util.func public @do_not_propagate_to_conv
//       APROP:   linalg.conv_2d_nhwc_hwcf

// CONV-LABEL:   util.func public @do_not_propagate_to_conv
//  CONV-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<18x2x18x8xf32>
//       CONV:   linalg.generic {{.*}} ins(%[[ARG0]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  util.func public @do_not_propagate_to_conv_generic(%arg0: tensor<18x2x18x8xf32>, %arg1: tensor<3x3x8x32xf32>) -> tensor<2x16x16x32xf32> {
    %0 = tensor.empty() : tensor<2x18x18x8xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<18x2x18x8xf32>) outs(%0 : tensor<2x18x18x8xf32>) permutation = [1, 0, 2, 3]
    %1 = tensor.empty() : tensor<2x16x16x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%transposed, %arg1 : tensor<2x18x18x8xf32>, tensor<3x3x8x32xf32>) outs(%1 : tensor<2x16x16x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<2x16x16x32xf32>
    util.return %2 : tensor<2x16x16x32xf32>
  }
}

// APROP-LABEL: util.func public @do_not_propagate_to_conv_generic
//       APROP:   linalg.transpose
//       APROP:   linalg.generic

// CONV-LABEL:   util.func public @do_not_propagate_to_conv_generic
//  CONV-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<18x2x18x8xf32>
//       CONV:   linalg.generic {{.*}} ins(%[[ARG0]]

// -----

util.func public @sink_through_expand_shape(%arg0: tensor<?x?x?xf32>) -> tensor<32x?x16x?x?xf32> {
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %0 = arith.divui %dim, %c16 : index
  %1 = arith.divui %dim_0, %c32 : index
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3], [4]] output_shape [16, %0, 32, %1, %dim_1] : tensor<?x?x?xf32> into tensor<16x?x32x?x?xf32>
  %dim_2 = tensor.dim %expanded, %c1 : tensor<16x?x32x?x?xf32>
  %dim_3 = tensor.dim %expanded, %c3 : tensor<16x?x32x?x?xf32>
  %dim_4 = tensor.dim %expanded, %c4 : tensor<16x?x32x?x?xf32>
  %2 = tensor.empty(%dim_3, %dim_2, %dim_4) : tensor<32x?x16x?x?xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<16x?x32x?x?xf32>) outs(%2 : tensor<32x?x16x?x?xf32>) permutation = [2, 3, 0, 1, 4]
  util.return %transposed : tensor<32x?x16x?x?xf32>
}
// SINK-LABEL: util.func public @sink_through_expand_shape
//       SINK:   %[[EXP:.+]] = tensor.expand_shape {{.*}} {{\[\[}}0, 1], [2, 3], [4]]
//  SINK-SAME:                   tensor<?x?x?xf32> into tensor<16x?x32x?x?xf32>
//       SINK:   %[[RES:.+]] = linalg.transpose ins(%[[EXP]] : tensor<16x?x32x?x?xf32>
//  SINK-SAME:                    outs({{.*}} : tensor<32x?x16x?x?xf32>)
//  SINK-SAME:                    permutation = [2, 3, 0, 1, 4]
//       SINK:   util.return %[[RES]] : tensor<32x?x16x?x?xf32>

// -----

util.func public @sink_non_involution_through_expand_shape(%arg0 : tensor<2x3x4xf32>) -> tensor<1x3x4x2xf32> {
  %empty = tensor.empty(): tensor<3x4x2xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>)
      outs(%empty : tensor<3x4x2xf32>) permutation = [1, 2, 0]
  %expanded = tensor.expand_shape %transposed [[0, 1], [2], [3]] output_shape [1, 3, 4, 2] : tensor<3x4x2xf32> into tensor<1x3x4x2xf32>
  util.return %expanded : tensor<1x3x4x2xf32>
}
// SINK-LABEL: util.func public @sink_non_involution_through_expand_shape
//       SINK:   %[[EXP:.+]] = tensor.expand_shape {{.*}} {{\[\[}}0], [1, 2], [3]]
//  SINK-SAME:                   tensor<2x3x4xf32> into tensor<2x1x3x4xf32>
//       SINK:   %[[RES:.+]] = linalg.transpose ins(%[[EXP]] : tensor<2x1x3x4xf32>
//  SINK-SAME:                    outs({{.*}} : tensor<1x3x4x2xf32>)
//  SINK-SAME:                    permutation = [1, 2, 3, 0]
//       SINK:   util.return %[[RES]] : tensor<1x3x4x2xf32>

// -----

util.func public @bubble_non_involution_through_collapse_shape(%arg0 : tensor<1x2x3x5x7x11xf32>) -> tensor<35x11x6xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3, 4], [5]] : tensor<1x2x3x5x7x11xf32> into tensor<6x35x11xf32>
  %empty = tensor.empty(): tensor<35x11x6xf32>
  %transposed = linalg.transpose ins(%collapsed : tensor<6x35x11xf32>)
      outs(%empty : tensor<35x11x6xf32>) permutation = [1, 2, 0]
  util.return %transposed : tensor<35x11x6xf32>
}
// BUBBLE-LABEL: util.func public @bubble_non_involution_through_collapse_shape
//       BUBBLE:   %[[T:.+]] = linalg.transpose ins(%{{.*}} : tensor<1x2x3x5x7x11xf32>
//  BUBBLE-SAME:                    outs({{.*}} : tensor<5x7x11x1x2x3xf32>)
//  BUBBLE-SAME:                    permutation = [3, 4, 5, 0, 1, 2]
//       BUBBLE:   %[[COL:.+]] = tensor.collapse_shape %[[T]] {{\[\[}}0, 1], [2], [3, 4, 5]]
//  BUBBLE-SAME:                   tensor<5x7x11x1x2x3xf32> into tensor<35x11x6xf32>
//       BUBBLE:   util.return %[[COL]] : tensor<35x11x6xf32>

// -----

util.func public @propagate_transpose_through_unary_elementwise(%arg0 : tensor<2x3x4xf32>) -> tensor<3x4x2xf32> {
  %empty = tensor.empty(): tensor<3x4x2xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>)
      outs(%empty : tensor<3x4x2xf32>) permutation = [1, 2, 0]
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%transposed : tensor<3x4x2xf32>)
                outs(%empty : tensor<3x4x2xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %sqrt = math.rsqrt %in : f32
                    linalg.yield %sqrt : f32
                  } -> tensor<3x4x2xf32>
  util.return %0 : tensor<3x4x2xf32>
}
// SINK-LABEL: util.func public @propagate_transpose_through_unary_elementwise
//       SINK:   %[[ELEM:.+]] = linalg.generic {{.*}} ins(%{{.*}} : tensor<2x3x4xf32>
//       SINK:     math.rsqrt
//       SINK:   %[[RES:.+]] = linalg.transpose ins(%[[ELEM]] : tensor<2x3x4xf32>
//  SINK-SAME:                    outs({{.*}} : tensor<3x4x2xf32>)
//  SINK-SAME:                    permutation = [1, 2, 0]
//       SINK:   util.return %[[RES]] : tensor<3x4x2xf32>

// -----

util.func public @propagate_transpose_up_through_unary_elementwise(%arg0 : tensor<2x3x4xf32>) -> tensor<3x4x2xf32> {
  %empty = tensor.empty(): tensor<2x3x4xf32>
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%arg0 : tensor<2x3x4xf32>)
                outs(%empty : tensor<2x3x4xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %sqrt = math.rsqrt %in : f32
                    linalg.yield %sqrt : f32
                  } -> tensor<2x3x4xf32>
  %empty1 = tensor.empty(): tensor<3x4x2xf32>
  %transposed = linalg.transpose ins(%0 : tensor<2x3x4xf32>)
      outs(%empty1 : tensor<3x4x2xf32>) permutation = [1, 2, 0]
  util.return %transposed : tensor<3x4x2xf32>
}
// BUBBLE-LABEL: util.func public @propagate_transpose_through_unary_elementwise
//       BUBBLE:   %[[TRANSPOSE:.+]] = linalg.transpose ins({{.*}} : tensor<2x3x4xf32>
//  BUBBLE-SAME:                    outs({{.*}} : tensor<3x4x2xf32>)
//  BUBBLE-SAME:                    permutation = [1, 2, 0]
//       BUBBLE:   %[[ELEM:.+]] = linalg.generic {{.*}} ins(%[[TRANSPOSE]] : tensor<3x4x2xf32>
//       BUBBLE:     math.rsqrt
//       BUBBLE:   util.return %[[ELEM]] : tensor<3x4x2xf32>

// -----

util.func public @sink_transpose_to_unary_transpose(%arg0 : tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %empty = tensor.empty(): tensor<3x4x2xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>)
      outs(%empty : tensor<3x4x2xf32>) permutation = [1, 2, 0]
  %empty2 = tensor.empty(): tensor<2x3x4xf32>
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d1, d2, d0)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%transposed : tensor<3x4x2xf32>)
                outs(%empty2 : tensor<2x3x4xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %sqrt = math.rsqrt %in : f32
                    linalg.yield %sqrt : f32
                  } -> tensor<2x3x4xf32>
  util.return %0 : tensor<2x3x4xf32>
}
// SINK-LABEL: util.func public @sink_transpose_to_unary_transpose
//  SINK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<2x3x4xf32>
//       SINK:   %[[ELEM:.+]] = linalg.generic
//  SINK-SAME:     ins(%[[ARG0]] : tensor<2x3x4xf32>
//  SINK-SAME:     outs(%{{.*}} : tensor<2x3x4xf32>
//       SINK:     math.rsqrt
//       SINK:   util.return %[[ELEM]] : tensor<2x3x4xf32>

// -----

util.func public @do_not_sink_multi_use(%arg0 : tensor<2x3x4xf32>) -> (tensor<3x4x2xf32>, tensor<3x4x2xf32>) {
  %empty = tensor.empty(): tensor<3x4x2xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>)
      outs(%empty : tensor<3x4x2xf32>) permutation = [1, 2, 0]
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%transposed : tensor<3x4x2xf32>)
                outs(%empty : tensor<3x4x2xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %sqrt = math.rsqrt %in : f32
                    linalg.yield %sqrt : f32
                  } -> tensor<3x4x2xf32>
  %1 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%transposed : tensor<3x4x2xf32>)
                outs(%empty : tensor<3x4x2xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %sqrt = math.exp %in : f32
                    linalg.yield %sqrt : f32
                  } -> tensor<3x4x2xf32>
  util.return %0, %1 : tensor<3x4x2xf32>, tensor<3x4x2xf32>
}
// SINK-LABEL: util.func public @do_not_sink_multi_use
//  SINK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<2x3x4xf32>
//       SINK:   %[[T:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<2x3x4xf32>
//  SINK-SAME:                    outs({{.*}} : tensor<3x4x2xf32>)
//  SINK-SAME:                    permutation = [1, 2, 0]
//       SINK:   %[[ELEM0:.+]] = linalg.generic {{.*}} ins(%[[T]]
//       SINK:     math.rsqrt
//       SINK:   %[[ELEM1:.+]] = linalg.generic {{.*}} ins(%[[T]]
//       SINK:     math.exp
//       SINK:   util.return %[[ELEM0]], %[[ELEM1]]

// -----

util.func public @do_not_sink_unary_multi_use(%arg0 : tensor<2x3x4xf32>) -> (tensor<3x4x2xf32>, tensor<3x4x2xf32>) {
  %empty = tensor.empty(): tensor<3x4x2xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>)
      outs(%empty : tensor<3x4x2xf32>) permutation = [1, 2, 0]
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%transposed : tensor<3x4x2xf32>)
                outs(%empty : tensor<3x4x2xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %sqrt = math.rsqrt %in : f32
                    linalg.yield %sqrt : f32
                  } -> tensor<3x4x2xf32>
  util.return %0, %transposed : tensor<3x4x2xf32>, tensor<3x4x2xf32>
}
// SINK-LABEL: util.func public @do_not_sink_unary_multi_use
//  SINK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<2x3x4xf32>
//       SINK:   %[[T:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<2x3x4xf32>
//  SINK-SAME:                    outs({{.*}} : tensor<3x4x2xf32>)
//  SINK-SAME:                    permutation = [1, 2, 0]
//       SINK:   %[[ELEM:.+]] = linalg.generic {{.*}} ins(%[[T]]
//       SINK:     math.rsqrt
//       SINK:   util.return %[[ELEM]], %[[T]]

// -----

util.func public @bubble_through_matmul(%lhs: tensor<16x16xf32>,
                                        %rhs: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %empty = tensor.empty(): tensor<16x16xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<16x16xf32>, tensor<16x16xf32>)
                            outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %transpose = linalg.transpose ins(%mm : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  util.return %transpose : tensor<16x16xf32>
}
//   APROP-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   APROP-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
//   APROP-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//       APROP: util.func public @bubble_through_matmul
//       APROP:   %[[EMPTY:.+]] = tensor.empty() : tensor<16x16xf32>
//       APROP:   %[[MATMUL:.+]] = linalg.generic
//  APROP-SAME:     indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
//  APROP-SAME:     outs(%[[EMPTY]] : tensor<16x16xf32>)
//       APROP:   util.return %[[MATMUL]]

// -----

util.func public @propagate_transpose_down_through_broadcast_elementwise(%arg0: tensor<3x4x2xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x3x4xf32> {
  %empty = tensor.empty(): tensor<2x3x4xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<3x4x2xf32>)
      outs(%empty : tensor<2x3x4xf32>) permutation = [2, 0, 1]
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%transposed, %arg1 : tensor<2x3x4xf32>, tensor<3x4xf32>)
                outs(%empty : tensor<2x3x4xf32>) {
                  ^bb0(%in: f32, %in1: f32, %out: f32):
                    %add = arith.addf %in, %in1 : f32
                    linalg.yield %add : f32
                  } -> tensor<2x3x4xf32>
  util.return %0 : tensor<2x3x4xf32>
}

//   SINK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   SINK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// SINK-LABEL: util.func public @propagate_transpose_down_through_broadcast_elementwise
//  SINK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<3x4x2xf32>
//  SINK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<3x4xf32>
//       SINK:   %[[ELEM:.+]] = linalg.generic
//  SINK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP]]]
//  SINK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<3x4x2xf32>, tensor<3x4xf32>
//       SINK:     arith.addf
//       SINK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[ELEM]] : tensor<3x4x2xf32>
//  SINK-SAME:                    outs({{.*}} : tensor<2x3x4xf32>)
//  SINK-SAME:                    permutation = [2, 0, 1]
//       SINK:   util.return %[[TRANSPOSE]] : tensor<2x3x4xf32>

// -----

util.func public @propagate_transpose_down_through_multi_operand_elementwise(%arg0: tensor<3x4x2xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x3x4xf32> {
  %empty = tensor.empty(): tensor<2x3x4xf32>
  %t1 = linalg.transpose ins(%arg0 : tensor<3x4x2xf32>)
      outs(%empty : tensor<2x3x4xf32>) permutation = [2, 0, 1]
  %empty2 = tensor.empty(): tensor<4x3xf32>
  %t2 = linalg.transpose ins(%arg1 : tensor<3x4xf32>)
      outs(%empty2 : tensor<4x3xf32>) permutation = [1, 0]
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d2, d1)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%t2, %t1 : tensor<4x3xf32>, tensor<2x3x4xf32>)
                outs(%empty : tensor<2x3x4xf32>) {
                  ^bb0(%in: f32, %in1: f32, %out: f32):
                    %add = arith.addf %in, %in1 : f32
                    linalg.yield %add : f32
                  } -> tensor<2x3x4xf32>
  util.return %0 : tensor<2x3x4xf32>
}

// Verify that it first selects the correct transpose to propagate and then
// fuses the transpose on the broadcasted operand.

//   SINK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//   SINK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// SINK-LABEL: util.func public @propagate_transpose_down_through_multi_operand_elementwise
//  SINK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<3x4x2xf32>
//  SINK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<3x4xf32>
//       SINK:   %[[ELEM:.+]] = linalg.generic
//  SINK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]]]
//  SINK-SAME:     ins(%[[ARG1]], %[[ARG0]] : tensor<3x4xf32>, tensor<3x4x2xf32>
//       SINK:     arith.addf
//       SINK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[ELEM]] : tensor<3x4x2xf32>
//  SINK-SAME:                    outs({{.*}} : tensor<2x3x4xf32>)
//  SINK-SAME:                    permutation = [2, 0, 1]
//       SINK:   util.return %[[TRANSPOSE]] : tensor<2x3x4xf32>

// -----

util.func public @sink_transpose_down_to_broadcast_elementwise(%arg0: tensor<3x4x2xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x3x4xf32> {
  %empty = tensor.empty(): tensor<2x3x4xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<3x4x2xf32>)
      outs(%empty : tensor<2x3x4xf32>) permutation = [2, 0, 1]
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%transposed, %arg1 : tensor<2x3x4xf32>, tensor<2x4xf32>)
                outs(%empty : tensor<2x3x4xf32>) {
                  ^bb0(%in: f32, %in1: f32, %out: f32):
                    %add = arith.addf %in, %in1 : f32
                    linalg.yield %add : f32
                  } -> tensor<2x3x4xf32>
  util.return %0 : tensor<2x3x4xf32>
}

// Verify that the transpose is fused rather than propagated because the
// broadcast operand would be affected by the transpose.

//   SINK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
//   SINK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   SINK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// SINK-LABEL: util.func public @sink_transpose_down_to_broadcast_elementwise
//  SINK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<3x4x2xf32>
//  SINK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<2x4xf32>
//       SINK:   %[[ELEM:.+]] = linalg.generic
//  SINK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  SINK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<3x4x2xf32>, tensor<2x4xf32>
//       SINK:     arith.addf
//       SINK:   util.return %[[ELEM]] : tensor<2x3x4xf32>

// -----

util.func public @propagate_transpose_up_through_broadcast_elementwise(%arg0: tensor<2x3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4x2xf32> {
  %empty = tensor.empty(): tensor<2x3x4xf32>
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%arg0, %arg1 : tensor<2x3x4xf32>, tensor<3x4xf32>)
                outs(%empty : tensor<2x3x4xf32>) {
                  ^bb0(%in: f32, %in1: f32, %out: f32):
                    %add = arith.addf %in, %in1 : f32
                    linalg.yield %add : f32
                  } -> tensor<2x3x4xf32>
  %empty1 = tensor.empty(): tensor<3x4x2xf32>
  %transposed = linalg.transpose ins(%0 : tensor<2x3x4xf32>)
      outs(%empty1 : tensor<3x4x2xf32>) permutation = [1, 2, 0]
  util.return %transposed : tensor<3x4x2xf32>
}

//   BUBBLE-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   BUBBLE-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// BUBBLE-LABEL: util.func public @propagate_transpose_up_through_broadcast_elementwise
//  BUBBLE-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<2x3x4xf32>
//  BUBBLE-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<3x4xf32>
//       BUBBLE:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<2x3x4xf32>
//  BUBBLE-SAME:                    outs({{.*}} : tensor<3x4x2xf32>)
//  BUBBLE-SAME:                    permutation = [1, 2, 0]
//       BUBBLE:   %[[ELEM:.+]] = linalg.generic
//  BUBBLE-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP]]]
//  BUBBLE-SAME:     ins(%[[TRANSPOSE]], %[[ARG1]] : tensor<3x4x2xf32>, tensor<3x4xf32>
//       BUBBLE:     arith.addf
//       BUBBLE:   util.return %[[ELEM]] : tensor<3x4x2xf32>

// -----

util.func public @bubble_transpose_to_broadcast_elementwise(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<3x4x2xf32> {
  %empty = tensor.empty(): tensor<2x3x4xf32>
  %0 = linalg.generic {indexing_maps = [
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%arg0, %arg1 : tensor<2x3x4xf32>, tensor<2x4xf32>)
                outs(%empty : tensor<2x3x4xf32>) {
                  ^bb0(%in: f32, %in1: f32, %out: f32):
                    %add = arith.addf %in, %in1 : f32
                    linalg.yield %add : f32
                  } -> tensor<2x3x4xf32>
  %empty1 = tensor.empty(): tensor<3x4x2xf32>
  %transposed = linalg.transpose ins(%0 : tensor<2x3x4xf32>)
      outs(%empty1 : tensor<3x4x2xf32>) permutation = [1, 2, 0]
  util.return %transposed : tensor<3x4x2xf32>
}

// Verify that the transpose is fused rather than propagated because the
// broadcast operand would be affected by the transpose.

//   BUBBLE-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
//   BUBBLE-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//   BUBBLE-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// BUBBLE-LABEL: util.func public @bubble_transpose_to_broadcast_elementwise
//  BUBBLE-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<2x3x4xf32>
//  BUBBLE-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<2x4xf32>
//       BUBBLE:   %[[ELEM:.+]] = linalg.generic
//  BUBBLE-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  BUBBLE-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<2x3x4xf32>, tensor<2x4xf32>
//       BUBBLE:     arith.addf
//       BUBBLE:   util.return %[[ELEM]] : tensor<3x4x2xf32>

// -----

util.func public @bubble_transpose_v_from_attention(%q: tensor<2x10x4096x64xf16>, %k: tensor<2x10x4096x64xf16>, %quantized_v: tensor<2x10x4096x64xi32>, %quant_offset: tensor<10x64xi32>, %quant_scale: tensor<10x64xf32>, %scale: f16) -> tensor<2x10x4096x64xf16> {
  // Dequantize int-quantization of V
  %init_dequant = tensor.empty() : tensor<2x10x4096x64xf16>
  %v = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%quantized_v, %quant_offset, %quant_scale : tensor<2x10x4096x64xi32>, tensor<10x64xi32>, tensor<10x64xf32>) outs(%init_dequant : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: i32, %in_0: i32, %in_1: f32, %out: f16):
      %19 = arith.addi %in, %in_0 : i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %in_1 : f32
      %22 = arith.truncf %21 : f32 to f16
      linalg.yield %22 : f16
  } -> tensor<2x10x4096x64xf16>

  // Attention with transposed V
  %init_attention = tensor.empty() : tensor<2x10x4096x64xf16>
  %attention = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%q, %k, %v, %scale : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, f16) outs(%init_attention : tensor<2x10x4096x64xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<2x10x4096x64xf16>
  util.return %attention : tensor<2x10x4096x64xf16>
}


// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
// CHECK-DAG: #[[$MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

// CHECK-LABEL:         util.func public @bubble_transpose_v_from_attention(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x10x4096x64xf16>, %[[ARG1:.*]]: tensor<2x10x4096x64xf16>, %[[ARG2:.*]]: tensor<2x10x4096x64xi32>,
// CHECK-SAME:         %[[ARG3:.*]]: tensor<10x64xi32>, %[[ARG4:.*]]: tensor<10x64xf32>, %[[ARG5:.*]]: f16) -> tensor<2x10x4096x64xf16> {
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<2x10x4096x64xf16>
// CHECK:         %[[DEQUANT_V:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[ARG2]], %[[ARG3]], %[[ARG4]] : tensor<2x10x4096x64xi32>, tensor<10x64xi32>, tensor<10x64xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<2x10x4096x64xf16>)
// CHECK:         %[[TRANS_V:.*]] = linalg.transpose ins(%[[DEQUANT_V]] : tensor<2x10x4096x64xf16>) outs({{.*}} : tensor<2x10x64x4096xf16>) permutation = [0, 1, 3, 2]
// CHECK:         %[[ATTN:.*]] = iree_linalg_ext.attention
// CHECK-SAME:    {indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_S]], #[[$MAP_O]]]}
// CHECK-SAME:    ins(%[[ARG0]], %[[ARG1]], %[[TRANS_V]], %[[ARG5]] : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x64x4096xf16>, f16)
// CHECK-SAME:    outs(%[[EMPTY]] : tensor<2x10x4096x64xf16>)
// CHECK:         util.return %[[ATTN]] : tensor<2x10x4096x64xf16>

// -----

util.func public @dont_reshape_reduction(%arg0: tensor<16x4x4xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %empty1 = tensor.empty(): tensor<16x4x4xf32>
  %0 = linalg.transpose ins(%arg0 : tensor<16x4x4xf32>)
      outs(%empty1 : tensor<16x4x4xf32>) permutation = [0, 2, 1]
  %collapse = tensor.collapse_shape %0 [[0], [1, 2]] : tensor<16x4x4xf32> into tensor<16x16xf32>
  %empty2 = tensor.empty(): tensor<16x16xf32>
  %1 = linalg.matmul ins(%collapse, %arg1: tensor<16x16xf32>, tensor<16x16xf32>)
                            outs(%empty2 : tensor<16x16xf32>) -> tensor<16x16xf32>

  util.return %1 : tensor<16x16xf32>
}
// APROP-LABEL: util.func public @dont_reshape_reduction
//       APROP:   %[[V0:.+]] = linalg.transpose
//       APROP:   %[[V1:.+]] = tensor.collapse_shape %[[V0]]
//       APROP:   %[[V2:.+]] = linalg.matmul ins(%[[V1]]
//       APROP:   util.return %[[V2]]
