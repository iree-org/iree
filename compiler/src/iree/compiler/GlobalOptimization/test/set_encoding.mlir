// RUN: iree-opt --iree-global-opt-set-encoding --cse --split-input-file %s | FileCheck %s

util.func public @matmul_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<100x250xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<250x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<250x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<100x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_f32f32f32_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[LHS_DIM0]]]
//      CHECK:   %[[LHS_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[LHS_DIM1]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[RHS_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[RHS_DIM0]]]
//      CHECK:   %[[RHS_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[RHS_DIM1]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_DIM0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[OUTS_DIM0]]]
//      CHECK:   %[[OUTS_DIM1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[OUTS_DIM1]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [{{.*}}] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_i8i8i32(%arg0 : tensor<100x250xi8>, %arg1 : tensor<250x500xi8>,
    %arg2 : tensor<100x500xi32>) -> tensor<100x500xi32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xi8>, tensor<250x500xi8>)
      outs(%arg2 : tensor<100x500xi32>) -> tensor<100x500xi32>
  util.return %0 : tensor<100x500xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xi32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xi8> to tensor<?x?xi8>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], original_type = tensor<100x250xi8>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xi8> to tensor<?x?xi8>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], original_type = tensor<250x500xi8>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xi32> to tensor<?x?xi32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], original_type = tensor<100x500xi32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_f16f16f32(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f16f16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xf16> to tensor<?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32], original_type = tensor<100x250xf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f32], original_type = tensor<250x500xf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], original_type = tensor<100x500xf32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_f16f16f16(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf16>) -> tensor<100x500xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf16>) -> tensor<100x500xf16>
  util.return %0 : tensor<100x500xf16>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f16f16f16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf16>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xf16> to tensor<?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], original_type = tensor<100x250xf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], original_type = tensor<250x500xf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], original_type = tensor<100x500xf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_bf16bf16f32(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], original_type = tensor<100x250xbf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], original_type = tensor<250x500xbf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], original_type = tensor<100x500xf32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_bf16bf16bf16(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16>
  util.return %0 : tensor<100x500xbf16>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xbf16>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], original_type = tensor<100x250xbf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], original_type = tensor<250x500xbf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], original_type = tensor<100x500xbf16>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_f32f32f32(%arg0 : tensor<64x100x250xf32>, %arg1 : tensor<64x250x500xf32>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf32>, tensor<64x250x500xf32>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  util.return %0 : tensor<64x100x500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<64x100x250xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<64x250x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<64x100x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_f32f32f32_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>,
    %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  util.return %0 : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>, %[[ARG2:.+]]: tensor<?x?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[LHS_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[LHS_DIM0]]]
//      CHECK:   %[[LHS_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[LHS_DIM1]]]
//      CHECK:   %[[LHS_DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[LHS_DIM2]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<?x?x?xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[RHS_DIM0]]]
//      CHECK:   %[[RHS_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[RHS_DIM1]]]
//      CHECK:   %[[RHS_DIM2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[RHS_DIM2]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<?x?x?xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[OUTS_DIM0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[OUTS_DIM0]]]
//      CHECK:   %[[OUTS_DIM1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[OUTS_DIM1]]]
//      CHECK:   %[[OUTS_DIM2:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[OUTS_DIM2]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<?x?x?xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [{{.*}}] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_f16f16f16(%arg0 : tensor<64x100x250xf16>, %arg1 : tensor<64x250x500xf16>,
    %arg2 : tensor<64x100x500xf16>) -> tensor<64x100x500xf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf16>, tensor<64x250x500xf16>)
      outs(%arg2 : tensor<64x100x500xf16>) -> tensor<64x100x500xf16>
  util.return %0 : tensor<64x100x500xf16>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f16f16f16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf16>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], original_type = tensor<64x100x250xf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], original_type = tensor<64x250x500xf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], original_type = tensor<64x100x500xf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_f16f16f32(%arg0 : tensor<64x100x250xf16>, %arg1 : tensor<64x250x500xf16>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf16>, tensor<64x250x500xf16>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  util.return %0 : tensor<64x100x500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f16f16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32], original_type = tensor<64x100x250xf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f32], original_type = tensor<64x250x500xf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], original_type = tensor<64x100x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_bf16bf16bf16(%arg0 : tensor<64x100x250xbf16>, %arg1 : tensor<64x250x500xbf16>,
    %arg2 : tensor<64x100x500xbf16>) -> tensor<64x100x500xbf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xbf16>, tensor<64x250x500xbf16>)
      outs(%arg2 : tensor<64x100x500xbf16>) -> tensor<64x100x500xbf16>
  util.return %0 : tensor<64x100x500xbf16>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xbf16>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], original_type = tensor<64x100x250xbf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], original_type = tensor<64x250x500xbf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], original_type = tensor<64x100x500xbf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_bf16bf16f32(%arg0 : tensor<64x100x250xbf16>, %arg1 : tensor<64x250x500xbf16>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xbf16>, tensor<64x250x500xbf16>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  util.return %0 : tensor<64x100x500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], original_type = tensor<64x100x250xbf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], original_type = tensor<64x250x500xbf16>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], original_type = tensor<64x100x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_i8i8i32(%arg0 : tensor<64x100x250xi8>, %arg1 : tensor<64x250x500xi8>,
    %arg2 : tensor<64x100x500xi32>) -> tensor<64x100x500xi32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xi8>, tensor<64x250x500xi8>)
      outs(%arg2 : tensor<64x100x500xi32>) -> tensor<64x100x500xi32>
  util.return %0 : tensor<64x100x500xi32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xi32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xi8> to tensor<?x?x?xi8>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], original_type = tensor<64x100x250xi8>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xi8> to tensor<?x?x?xi8>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], original_type = tensor<64x250x500xi8>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xi32> to tensor<?x?x?xi32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], original_type = tensor<64x100x500xi32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @vecmat_f32f32f32(%arg0 : tensor<250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<500xf32>) -> tensor<500xf32> {
  %0 = linalg.vecmat ins(%arg0, %arg1 : tensor<250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<500xf32>) -> tensor<500xf32>
  util.return %0 : tensor<500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: util.func public @vecmat_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<500xf32>
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]] = iree_linalg_ext.upper_bound_tile_size tensor<250xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index
//      CHECK:   %[[LHS_PADDING_SIZE:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]], %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0] high[%[[LHS_PADDING_SIZE]]]
//      CHECK:       tensor<250xf32> to tensor<?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<250xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<250x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<250x500xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]] = iree_linalg_ext.upper_bound_tile_size tensor<500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]], %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0] high[%[[OUTS_PADDING_SIZE]]]
//      CHECK:       tensor<500xf32> to tensor<?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<500xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[VECMAT:.+]] = linalg.vecmat
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[VECMAT]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0] [500] [1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matvec_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  util.return %0 : tensor<100xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: util.func public @matvec_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100xf32>
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<100x250xf32>, matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]] = iree_linalg_ext.upper_bound_tile_size tensor<250xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index
//      CHECK:   %[[RHS_PADDING_SIZE:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]], %[[C250]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0] high[%[[RHS_PADDING_SIZE]]]
//      CHECK:       tensor<250xf32> to tensor<?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<250xf32>, matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]] = iree_linalg_ext.upper_bound_tile_size tensor<100xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]], %[[C100]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0] high[%[[OUTS_PADDING_SIZE]]]
//      CHECK:       tensor<100xf32> to tensor<?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<100xf32>, matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[MATVEC:.+]] = linalg.matvec
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATVEC]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0] [100] [1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_vecmat_f32f32f32(%arg0 : tensor<3x250xf32>, %arg1 : tensor<3x250x500xf32>,
    %arg2 : tensor<3x500xf32>) -> tensor<3x500xf32> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<3x250xf32>, tensor<3x250x500xf32>)
      outs(%arg2 : tensor<3x500xf32>) -> tensor<3x500xf32>
  util.return %0 : tensor<3x500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @batch_vecmat_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<3x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<3x250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<3x500xf32>
//  CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<3x250xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C3]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<3x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<3x250xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<3x250x500xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C3]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<3x250x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<3x250x500xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<3x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C3]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<3x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<3x500xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[VECMAT:.+]] = linalg.batch_vecmat
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[VECMAT]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [3, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matvec_f32f32f32_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.batch_matvec ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @batch_matvec_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[LHS_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[LHS_DIM0]]]
//      CHECK:   %[[LHS_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[LHS_DIM1]]]
//      CHECK:   %[[LHS_DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[LHS_DIM2]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<?x?x?xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[RHS_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[RHS_DIM0]]]
//      CHECK:   %[[RHS_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[RHS_DIM1]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_DIM0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[OUTS_DIM0]]]
//      CHECK:   %[[OUTS_DIM1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[OUTS_DIM1]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATVEC:.+]] = linalg.batch_matvec
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATVEC]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [{{.*}}] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @fold_fill_with_set_encoding(%arg0 : index, %arg1 : index)
  -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = iree_linalg_ext.set_encoding %1 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
  util.return %2 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
}
//      CHECK: util.func public @fold_fill_with_set_encoding(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>)
//      CHECK:   util.return %[[FILL]]

// -----

util.func public @fold_fill_with_tensor_pad(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.pad %1 low[0, 0] high[%arg2, %arg3] {
  ^bb0(%b0: index, %b1 : index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = iree_linalg_ext.set_encoding %2 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>>
  util.return %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>>
}
//      CHECK: util.func public @fold_fill_with_tensor_pad(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[FILL]]

// -----

#compilation0 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0]]>,
    translation_info  = <CPUDefault>>

#compilation1 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0]]>,
    translation_info  = <CPUDefault>>


util.func public @preset_compilation_info(
    %arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>,
    %arg3 : tensor<?x?x?xf32>,
    %arg4 : tensor<?x?x?xf32>,
    %arg5 : tensor<?x?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?x?xf32>) {
  %0 = linalg.matmul {compilation_info = #compilation0} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.batch_matmul {compilation_info = #compilation1} ins(%arg3, %arg4 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%arg5 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  util.return %0, %1 : tensor<?x?xf32>, tensor<?x?x?xf32>
}
// CHECK-LABEL: util.func public @preset_compilation_info
// CHECK-NOT:     set_encoding
// CHECK-NOT:     unset_encoding
// CHECK:         linalg.matmul
// CHECK:         linalg.batch_matmul

// -----

util.func public @batch_matmul_truncf_f16f16f32(%arg0 : tensor<64x100x250xf32>, %arg1 : tensor<64x250x500xf32>,
      %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = tensor.empty() : tensor<64x250x500xf16>
  %casted0 = arith.truncf %arg0 : tensor<64x100x250xf32> to tensor<64x100x250xf16>
  %casted1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                              iterator_types = ["parallel", "parallel", "parallel"]}
                              ins(%arg1 : tensor<64x250x500xf32>)
                              outs(%0 : tensor<64x250x500xf16>) {
  ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
  } -> tensor<64x250x500xf16>
  %1 = linalg.batch_matmul ins(%casted0, %casted1 : tensor<64x100x250xf16>, tensor<64x250x500xf16>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  util.return %1 : tensor<64x100x500xf32>
}

//      CHECK: util.func public @batch_matmul_truncf_f16f16f32(%[[ARG0:.+]]: tensor<64x100x250xf32>, %[[ARG1:.+]]: tensor<64x250x500xf32>
//  CHECK-DAG: %[[INIT:.+]] = tensor.empty() : tensor<64x250x500xf16>
//  CHECK-DAG: arith.truncf %[[ARG0]] : tensor<64x100x250xf32> to tensor<64x100x250xf16>
//      CHECK: linalg.generic
// CHECK-SAME:   ins(%[[ARG1]] : tensor<64x250x500xf32>)
// CHECK-SAME:   outs(%[[INIT]] : tensor<64x250x500xf16>)
//      CHECK: element_types = [f16, f16, f32]

// -----

util.func public @matmul_casted_from_i1_f32f32f32(%arg0 : tensor<64x256xi1>,
    %arg1 : tensor<256x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %casted = arith.uitofp %arg0 : tensor<64x256xi1> to tensor<64x256xf32>
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = linalg.matmul ins(%casted, %arg1 : tensor<64x256xf32>, tensor<256x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  util.return %2 : tensor<64x128xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_casted_from_i1_f32f32f32
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<64x256xf32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<256x128xf32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<64x128xf32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>

// -----

util.func public @matmul_generic_casted_from_i1_f32f32f32(%arg0 : tensor<64x256xi1>,
    %arg1 : tensor<256x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<64x256xf32>
  %casted = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                              affine_map<(d0, d1) -> (d0, d1)>],
                              iterator_types = ["parallel", "parallel"]}
                              ins(%arg0 : tensor<64x256xi1>)
                              outs(%init : tensor<64x256xf32>) {
  ^bb0(%in: i1, %out: f32):
      %1 = arith.uitofp %in : i1 to f32
      linalg.yield %1 : f32
  } -> tensor<64x256xf32>
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = linalg.matmul ins(%casted, %arg1 : tensor<64x256xf32>, tensor<256x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  util.return %2 : tensor<64x128xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_generic_casted_from_i1_f32f32f32
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<64x256xf32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<256x128xf32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<64x128xf32>, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>

// -----

util.func public @matmul_f32f32f32_narrow_M(%arg0 : tensor<2x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<2x500xf32>) -> tensor<2x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<2x500xf32>) -> tensor<2x500xf32>
  util.return %0 : tensor<2x500xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f32f32f32_narrow_M(
//      CHECK:  iree_linalg_ext.upper_bound_tile_size tensor<2x250xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], matmul_narrow_M = 2 : index, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:  iree_linalg_ext.upper_bound_tile_size tensor<250x500xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], matmul_narrow_M = 2 : index, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:  iree_linalg_ext.upper_bound_tile_size tensor<2x500xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 2 : index, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   linalg.matmul

// -----

util.func public @batch_matmul_f32f32f32_narrow_MN(%arg0 : tensor<64x4x250xf32>, %arg1 : tensor<64x250x2xf32>,
    %arg2 : tensor<64x4x2xf32>) -> tensor<64x4x2xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x4x250xf32>, tensor<64x250x2xf32>)
      outs(%arg2 : tensor<64x4x2xf32>) -> tensor<64x4x2xf32>
  util.return %0 : tensor<64x4x2xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f32f32f32_narrow_MN(
//      CHECK:   iree_linalg_ext.upper_bound_tile_size tensor<64x4x250xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], matmul_narrow_M = 4 : index, matmul_narrow_N = 2 : index, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   iree_linalg_ext.upper_bound_tile_size tensor<64x250x2xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], matmul_narrow_M = 4 : index, matmul_narrow_N = 2 : index, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   iree_linalg_ext.upper_bound_tile_size tensor<64x4x2xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 4 : index, matmul_narrow_N = 2 : index, user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>>
//      CHECK:   linalg.batch_matmul

// -----

util.func public @matmul_transpose_a_f32f32f32(%arg0 : tensor<250x100xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<250x100xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_transpose_a_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<250x100xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x100xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C250]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<250x100xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<250x100xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<250x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<250x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<100x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul_transpose_a
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_transpose_b_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<500x250xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<500x250xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_transpose_b_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<500x250xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<100x250xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<500x250xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C500]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<500x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<500x250xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<100x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul_transpose_b
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_transpose_a_f32f32f32(%arg0 : tensor<2x250x100xf32>, %arg1 : tensor<2x250x500xf32>,
    %arg2 : tensor<2x100x500xf32>) -> tensor<2x100x500xf32> {
  %0 = linalg.batch_matmul_transpose_a ins(%arg0, %arg1 : tensor<2x250x100xf32>, tensor<2x250x500xf32>)
      outs(%arg2 : tensor<2x100x500xf32>) -> tensor<2x100x500xf32>
  util.return %0 : tensor<2x100x500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_transpose_a_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x250x100xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2x250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<2x100x500xf32>
//  CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x250x100xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C2]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C100]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<2x250x100xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<2x250x100xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x250x500xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C2]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<2x250x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<2x250x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C2]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<2x100x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<2x100x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul_transpose_a
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [2, 100, 500] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_transpose_b_f32f32f32(%arg0 : tensor<2x100x250xf32>, %arg1 : tensor<2x500x250xf32>,
    %arg2 : tensor<2x100x500xf32>) -> tensor<2x100x500xf32> {
  %0 = linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<2x100x250xf32>, tensor<2x500x250xf32>)
      outs(%arg2 : tensor<2x100x500xf32>) -> tensor<2x100x500xf32>
  util.return %0 : tensor<2x100x500xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_transpose_b_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2x500x250xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<2x100x500xf32>
//  CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x100x250xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C2]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<2x100x250xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<2x100x250xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x500x250xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C2]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<2x500x250xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<2x500x250xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x100x500xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C2]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<2x100x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<2x100x500xf32>, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul_transpose_b
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [2, 100, 500] [1, 1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @generic_batch_vecmat_transposed_i16u4i32(%arg0 : tensor<32x128xi16>, %arg1 : tensor<4096x32x128xi4>,
    %arg2 : tensor<4096x32xi32>) -> tensor<4096x32xi32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x128xi16>, tensor<4096x32x128xi4>) outs(%arg2 : tensor<4096x32xi32>) {
  ^bb0(%in: i16, %in_5: i4, %out: i32):
    %22 = arith.extsi %in : i16 to i32
    %23 = arith.extui %in_5 : i4 to i32
    %24 = arith.muli %22, %23 : i32
    %25 = arith.addi %24, %out : i32
    linalg.yield %25 : i32
  } -> tensor<4096x32xi32>
  util.return %0 : tensor<4096x32xi32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @generic_batch_vecmat_transposed_i16u4i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<32x128xi16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<4096x32x128xi4>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<4096x32xi32>
//  CHECK-DAG:     %[[C4096:.+]] = arith.constant 4096 : index
//  CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<32x128xi16, #iree_linalg_ext.encoding<role =  LHS, element_types = [i16, ui4, i32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C32]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C128]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<32x128xi16> to tensor<?x?xi16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xi16, #iree_linalg_ext.encoding<role =  LHS, element_types = [i16, ui4, i32], original_type = tensor<32x128xi16>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<4096x32x128xi4, #iree_linalg_ext.encoding<role =  RHS, element_types = [i16, ui4, i32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C4096]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C32]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C128]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<4096x32x128xi4> to tensor<?x?x?xi4>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi4, #iree_linalg_ext.encoding<role =  RHS, element_types = [i16, ui4, i32], original_type = tensor<4096x32x128xi4>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<4096x32xi32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [i16, ui4, i32], matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C4096]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C32]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<4096x32xi32> to tensor<?x?xi32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xi32> -> tensor<?x?xi32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [i16, ui4, i32], original_type = tensor<4096x32xi32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>>
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[GENERIC]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [4096, 32] [1, 1]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @dot(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<f32> {
  %res = "stablehlo.dot"(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<f32>
  util.return %res : tensor<f32>
}

// CHECK: util.func public @dot(
// CHECK: stablehlo.dot %{{.*}}, %{{.*}} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<f32>

// -----

util.func public @multi_m_dim_generic(%arg0 : tensor<64x4x128xf32>, %arg1 : tensor<128x512xf32>,
    %arg2 : tensor<64x4x512xf32>) -> tensor<64x4x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1 : tensor<64x4x128xf32>, tensor<128x512xf32>) outs(%arg2 : tensor<64x4x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<64x4x512xf32>
  util.return %4 : tensor<64x4x512xf32>
}

//      CHECK: util.func public @multi_m_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<64x4x128xf32>, tensor<128x512xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<64x4x512xf32>)

// -----

util.func public @multi_n_dim_generic(%arg0 : tensor<256x128xf32>, %arg1 : tensor<128x64x8xf32>,
    %arg2 : tensor<256x64x8xf32>) -> tensor<256x64x8xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1 : tensor<256x128xf32>, tensor<128x64x8xf32>) outs(%arg2 : tensor<256x64x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<256x64x8xf32>
  util.return %4 : tensor<256x64x8xf32>
}

//      CHECK: util.func public @multi_n_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<256x128xf32>, tensor<128x64x8xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<256x64x8xf32>)

// -----

util.func public @multi_k_dim_generic(%arg0 : tensor<256x64x2xf32>, %arg1 : tensor<64x2x512xf32>,
    %arg2 : tensor<256x512xf32>) -> tensor<256x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1 : tensor<256x64x2xf32>, tensor<64x2x512xf32>) outs(%arg2 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<256x512xf32>
  util.return %4 : tensor<256x512xf32>
}

//      CHECK: util.func public @multi_k_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<256x64x2xf32>, tensor<64x2x512xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<256x512xf32>)

// -----

util.func public @multi_batch_dim_generic(%arg0 : tensor<4x8x256x128xf32>, %arg1 : tensor<4x8x128x512xf32>,
    %arg2 : tensor<4x8x256x512xf32>) -> tensor<4x8x256x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1 : tensor<4x8x256x128xf32>, tensor<4x8x128x512xf32>) outs(%arg2 : tensor<4x8x256x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<4x8x256x512xf32>
  util.return %4 : tensor<4x8x256x512xf32>
}

//      CHECK: util.func public @multi_batch_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<4x8x256x128xf32>, tensor<4x8x128x512xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<4x8x256x512xf32>)
