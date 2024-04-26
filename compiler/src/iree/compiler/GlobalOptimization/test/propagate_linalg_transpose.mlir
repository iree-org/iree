// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose{enable-aggressive-propagation=true}))" --split-input-file %s | FileCheck %s --check-prefix=APROP
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose{test-sinking-only=true}))" --split-input-file %s | FileCheck %s --check-prefix=SINK
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-propagate-linalg-transpose{test-bubbling-only=true}))" --split-input-file %s | FileCheck %s --check-prefix=BUBBLE

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
// CHECK-LABEL: util.func public @propagate_to_matmul_ops
//       CHECK:   linalg.matmul_transpose_b
//       CHECK:   %[[SECOND_MM:.+]] = linalg.matmul_transpose_a
//       CHECK:   util.return %[[SECOND_MM]]

// -----

util.func public @propagate_to_transposed_matmul_ops(%lhs: tensor<16x16xf32>,
                                              %second_lhs: tensor<16x16xf32>,
                                              %rhs: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %empty = tensor.empty(): tensor<16x16xf32>
  %transpose_b = linalg.transpose ins(%rhs : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  %first_mm = linalg.matmul_transpose_b ins(%lhs, %transpose_b : tensor<16x16xf32>, tensor<16x16xf32>)
                                        outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>

  %transpose_a = linalg.transpose ins(%second_lhs : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  %second_mm = linalg.matmul_transpose_a ins(%transpose_a, %first_mm : tensor<16x16xf32>, tensor<16x16xf32>)
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
// CHECK-LABEL: util.func public @propagate_to_bmm_ops
//       CHECK:   linalg.batch_matmul_transpose_b
//       CHECK:   %[[SECOND_MM:.+]] = linalg.batch_matmul_transpose_a
//       CHECK:   util.return %[[SECOND_MM]]

// -----

util.func public @propagate_to_transposed_bmm_ops(%lhs: tensor<2x16x16xf32>,
                                              %second_lhs: tensor<2x16x16xf32>,
                                              %rhs: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %empty = tensor.empty(): tensor<2x16x16xf32>
  %transpose_b = linalg.transpose ins(%rhs : tensor<2x16x16xf32>)
      outs(%empty : tensor<2x16x16xf32>) permutation = [0, 2, 1]
  %first_bmm = linalg.batch_matmul_transpose_b ins(%lhs, %transpose_b : tensor<2x16x16xf32>, tensor<2x16x16xf32>)
                                        outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

  %transpose_a = linalg.transpose ins(%second_lhs : tensor<2x16x16xf32>)
      outs(%empty : tensor<2x16x16xf32>) permutation = [0, 2, 1]
  %second_bmm = linalg.batch_matmul_transpose_a ins(%transpose_a, %first_bmm : tensor<2x16x16xf32>, tensor<2x16x16xf32>)
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

// -----

util.func public @sink_through_expand_shape(%arg0 : tensor<?x?x?xf32>) -> tensor<32x?x16x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %empty = tensor.empty(%d0, %d1, %d2): tensor<?x?x?xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<?x?x?xf32>)
      outs(%empty : tensor<?x?x?xf32>) permutation = [1, 0, 2]
  %expanded = tensor.expand_shape %transposed [[0, 1], [2, 3], [4]] : tensor<?x?x?xf32> into tensor<32x?x16x?x?xf32>
  util.return %expanded : tensor<32x?x16x?x?xf32>
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
  %expanded = tensor.expand_shape %transposed [[0, 1], [2], [3]] : tensor<3x4x2xf32> into tensor<1x3x4x2xf32>
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
//       APROP:   %[[MATMUL:.+]] = linalg.generic
//       APROP:     indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
//       APROP:   util.return %[[MATMUL]]
