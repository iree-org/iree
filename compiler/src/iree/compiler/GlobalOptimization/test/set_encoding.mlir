// RUN: iree-opt --iree-global-opt-set-encoding --cse --split-input-file %s | FileCheck %s

func.func @matmul_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<100x250xf32>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<250x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<250x500xf32>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f32f32f32_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @matmul_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> index, index
//      CHECK:   %[[LHS_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[LHS_DIM0]]]
//      CHECK:   %[[LHS_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[LHS_DIM1]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>> -> index, index
//      CHECK:   %[[RHS_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[RHS_DIM0]]]
//      CHECK:   %[[RHS_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[RHS_DIM1]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> index, index
//      CHECK:   %[[OUTS_DIM0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[OUTS_DIM0]]]
//      CHECK:   %[[OUTS_DIM1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[OUTS_DIM1]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [{{.*}}] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_i8i8i32(%arg0 : tensor<100x250xi8>, %arg1 : tensor<250x500xi8>,
    %arg2 : tensor<100x500xi32>) -> tensor<100x500xi32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xi8>, tensor<250x500xi8>)
      outs(%arg2 : tensor<100x500xi32>) -> tensor<100x500xi32>
  return %0 : tensor<100x500xi32>
}
//      CHECK: func @matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xi32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xi8, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [i8, i8, i32]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xi8> to tensor<?x?xi8>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [i8, i8, i32], original_type = tensor<100x250xi8>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xi8, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [i8, i8, i32]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xi8> to tensor<?x?xi8>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [i8, i8, i32], original_type = tensor<250x500xi8>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xi32> to tensor<?x?xi32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32], original_type = tensor<100x500xi32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f16f16f32(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: func @matmul_f16f16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf16, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f16, f16, f32]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xf16> to tensor<?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f16, f16, f32], original_type = tensor<100x250xf16>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf16, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f16, f16, f32]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f16, f16, f32], original_type = tensor<250x500xf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f16, f16, f32]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f16, f16, f32], original_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f16f16f16(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf16>) -> tensor<100x500xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf16>) -> tensor<100x500xf16>
  return %0 : tensor<100x500xf16>
}
//      CHECK: func @matmul_f16f16f16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf16>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf16, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f16, f16, f16]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xf16> to tensor<?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f16, f16, f16], original_type = tensor<100x250xf16>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf16, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f16, f16, f16]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f16, f16, f16], original_type = tensor<250x500xf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf16, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f16, f16, f16]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f16, f16, f16], original_type = tensor<100x500xf16>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_bf16bf16f32(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: func @matmul_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [bf16, bf16, f32]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [bf16, bf16, f32], original_type = tensor<100x250xbf16>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [bf16, bf16, f32]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [bf16, bf16, f32], original_type = tensor<250x500xbf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [bf16, bf16, f32]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [bf16, bf16, f32], original_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_bf16bf16bf16(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16>
  return %0 : tensor<100x500xbf16>
}
//      CHECK: func @matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xbf16>
//      CHECK:   %[[LHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [bf16, bf16, bf16]>> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high
//      CHECK:       tensor<100x250xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [bf16, bf16, bf16], original_type = tensor<100x250xbf16>>>
//      CHECK:   %[[RHS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [bf16, bf16, bf16]>> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high
//      CHECK:       tensor<250x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [bf16, bf16, bf16], original_type = tensor<250x500xbf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x500xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [bf16, bf16, bf16]>> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high
//      CHECK:       tensor<100x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [bf16, bf16, bf16], original_type = tensor<100x500xbf16>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_matmul_f32f32f32(%arg0 : tensor<64x100x250xf32>, %arg1 : tensor<64x250x500xf32>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf32>, tensor<64x250x500xf32>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  return %0 : tensor<64x100x500xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<64x100x250xf32>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<64x250x500xf32>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<64x100x500xf32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_matmul_f32f32f32_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>,
    %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_matmul_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>, %[[ARG2:.+]]: tensor<?x?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> index, index, index
//      CHECK:   %[[LHS_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[LHS_DIM0]]]
//      CHECK:   %[[LHS_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[LHS_DIM1]]]
//      CHECK:   %[[LHS_DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[LHS_DIM2]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<?x?x?xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>> -> index, index, index
//      CHECK:   %[[RHS_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[RHS_DIM0]]]
//      CHECK:   %[[RHS_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[RHS_DIM1]]]
//      CHECK:   %[[RHS_DIM2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[RHS_DIM2]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<?x?x?xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> index, index, index
//      CHECK:   %[[OUTS_DIM0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[OUTS_DIM0]]]
//      CHECK:   %[[OUTS_DIM1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[OUTS_DIM1]]]
//      CHECK:   %[[OUTS_DIM2:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[OUTS_DIM2]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<?x?x?xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [{{.*}}] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_matmul_f16f16f16(%arg0 : tensor<64x100x250xf16>, %arg1 : tensor<64x250x500xf16>,
    %arg2 : tensor<64x100x500xf16>) -> tensor<64x100x500xf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf16>, tensor<64x250x500xf16>)
      outs(%arg2 : tensor<64x100x500xf16>) -> tensor<64x100x500xf16>
  return %0 : tensor<64x100x500xf16>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_matmul_f16f16f16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf16>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f16, f16, f16]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f16, f16, f16], original_type = tensor<64x100x250xf16>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f16, f16, f16]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f16, f16, f16], original_type = tensor<64x250x500xf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f16, f16, f16]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f16, f16, f16], original_type = tensor<64x100x500xf16>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_matmul_f16f16f32(%arg0 : tensor<64x100x250xf16>, %arg1 : tensor<64x250x500xf16>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf16>, tensor<64x250x500xf16>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  return %0 : tensor<64x100x500xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_matmul_f16f16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f16, f16, f32]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f16, f16, f32], original_type = tensor<64x100x250xf16>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f16, f16, f32]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xf16> to tensor<?x?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f16, f16, f32], original_type = tensor<64x250x500xf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f16, f16, f32]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f16, f16, f32], original_type = tensor<64x100x500xf32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_matmul_bf16bf16bf16(%arg0 : tensor<64x100x250xbf16>, %arg1 : tensor<64x250x500xbf16>,
    %arg2 : tensor<64x100x500xbf16>) -> tensor<64x100x500xbf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xbf16>, tensor<64x250x500xbf16>)
      outs(%arg2 : tensor<64x100x500xbf16>) -> tensor<64x100x500xbf16>
  return %0 : tensor<64x100x500xbf16>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xbf16>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [bf16, bf16, bf16]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [bf16, bf16, bf16], original_type = tensor<64x100x250xbf16>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [bf16, bf16, bf16]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [bf16, bf16, bf16], original_type = tensor<64x250x500xbf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [bf16, bf16, bf16]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [bf16, bf16, bf16], original_type = tensor<64x100x500xbf16>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_matmul_bf16bf16f32(%arg0 : tensor<64x100x250xbf16>, %arg1 : tensor<64x250x500xbf16>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xbf16>, tensor<64x250x500xbf16>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  return %0 : tensor<64x100x500xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_matmul_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [bf16, bf16, f32]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [bf16, bf16, f32], original_type = tensor<64x100x250xbf16>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [bf16, bf16, f32]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xbf16> to tensor<?x?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xbf16, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [bf16, bf16, f32], original_type = tensor<64x250x500xbf16>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [bf16, bf16, f32]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [bf16, bf16, f32], original_type = tensor<64x100x500xf32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_matmul_i8i8i32(%arg0 : tensor<64x100x250xi8>, %arg1 : tensor<64x250x500xi8>,
    %arg2 : tensor<64x100x500xi32>) -> tensor<64x100x500xi32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xi8>, tensor<64x250x500xi8>)
      outs(%arg2 : tensor<64x100x500xi32>) -> tensor<64x100x500xi32>
  return %0 : tensor<64x100x500xi32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xi32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [i8, i8, i32]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xi8> to tensor<?x?x?xi8>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [i8, i8, i32], original_type = tensor<64x100x250xi8>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [i8, i8, i32]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xi8> to tensor<?x?x?xi8>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [i8, i8, i32], original_type = tensor<64x250x500xi8>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [i8, i8, i32]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xi32> to tensor<?x?x?xi32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [i8, i8, i32], original_type = tensor<64x100x500xi32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @vecmat_f32f32f32(%arg0 : tensor<250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<500xf32>) -> tensor<500xf32> {
  %0 = linalg.vecmat ins(%arg0, %arg1 : tensor<250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<500xf32>) -> tensor<500xf32>
  return %0 : tensor<500xf32>
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @vecmat_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<500xf32>
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]] = iree_linalg_ext.upper_bound_tile_size tensor<250xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index>> -> index
//      CHECK:   %[[LHS_PADDING_SIZE:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]], %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0] high[%[[LHS_PADDING_SIZE]]]
//      CHECK:       tensor<250xf32> to tensor<?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<250xf32>, matmul_narrow_M = 1 : index>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<250x500xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index>> -> index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<250x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<250x500xf32>, matmul_narrow_M = 1 : index>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]] = iree_linalg_ext.upper_bound_tile_size tensor<500xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index>> -> index
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]], %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0] high[%[[OUTS_PADDING_SIZE]]]
//      CHECK:       tensor<500xf32> to tensor<?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<500xf32>, matmul_narrow_M = 1 : index>>
//      CHECK:   %[[VECMAT:.+]] = linalg.vecmat
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[VECMAT]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0] [500] [1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matvec_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  return %0 : tensor<100xf32>
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @matvec_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100xf32>
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<100x250xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<100x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<100x250xf32>, matmul_narrow_N = 1 : index>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]] = iree_linalg_ext.upper_bound_tile_size tensor<250xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>> -> index
//      CHECK:   %[[RHS_PADDING_SIZE:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]], %[[C250]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0] high[%[[RHS_PADDING_SIZE]]]
//      CHECK:       tensor<250xf32> to tensor<?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<250xf32>, matmul_narrow_N = 1 : index>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]] = iree_linalg_ext.upper_bound_tile_size tensor<100xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>> -> index
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]], %[[C100]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0] high[%[[OUTS_PADDING_SIZE]]]
//      CHECK:       tensor<100xf32> to tensor<?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<100xf32>, matmul_narrow_N = 1 : index>>
//      CHECK:   %[[MATVEC:.+]] = linalg.matvec
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATVEC]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0] [100] [1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_vecmat_f32f32f32(%arg0 : tensor<3x250xf32>, %arg1 : tensor<3x250x500xf32>,
    %arg2 : tensor<3x500xf32>) -> tensor<3x500xf32> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<3x250xf32>, tensor<3x250x500xf32>)
      outs(%arg2 : tensor<3x500xf32>) -> tensor<3x500xf32>
  return %0 : tensor<3x500xf32>
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_vecmat_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<3x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<3x250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<3x500xf32>
//  CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<3x250xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index>> -> index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C3]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]]]
//      CHECK:       tensor<3x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<3x250xf32>, matmul_narrow_M = 1 : index>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<3x250x500xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C3]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<3x250x500xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<3x250x500xf32>, matmul_narrow_M = 1 : index>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<3x500xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index>> -> index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C3]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<3x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<3x500xf32>, matmul_narrow_M = 1 : index>>
//      CHECK:   %[[VECMAT:.+]] = linalg.batch_vecmat
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[VECMAT]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [3, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @batch_matvec_f32f32f32_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.batch_matvec ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//      CHECK: func @batch_matvec_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>> -> index, index, index
//      CHECK:   %[[LHS_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[LHS_DIM0]]]
//      CHECK:   %[[LHS_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[LHS_DIM1]]]
//      CHECK:   %[[LHS_DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[LHS_DIM2]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<?x?x?xf32> to tensor<?x?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>> -> index, index
//      CHECK:   %[[RHS_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[RHS_DIM0]]]
//      CHECK:   %[[RHS_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[RHS_DIM1]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>> -> index, index
//      CHECK:   %[[OUTS_DIM0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[OUTS_DIM0]]]
//      CHECK:   %[[OUTS_DIM1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[OUTS_DIM1]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]]]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index>>
//      CHECK:   %[[BATCH_MATVEC:.+]] = linalg.batch_matvec
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATVEC]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [{{.*}}] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @fold_fill_with_set_encoding(%arg0 : index, %arg1 : index)
  -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = iree_linalg_ext.set_encoding %1 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  return %2 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
}
//      CHECK: func @fold_fill_with_set_encoding(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>)
//      CHECK:   return %[[FILL]]

// -----

func.func @fold_fill_with_tensor_pad(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.pad %1 low[0, 0] high[%arg2, %arg3] {
  ^bb0(%b0: index, %b1 : index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = iree_linalg_ext.set_encoding %2 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  return %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
}
//      CHECK: func @fold_fill_with_tensor_pad(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   return %[[FILL]]

// -----

#compilation0 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0]]>,
    translation_info  = <CPUDefault>>

#compilation1 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0]]>,
    translation_info  = <CPUDefault>>


func.func @preset_compilation_info(
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
  return %0, %1 : tensor<?x?xf32>, tensor<?x?x?xf32>
}
// CHECK-LABEL: func.func @preset_compilation_info
// CHECK-NOT:     set_encoding
// CHECK-NOT:     unset_encoding
// CHECK:         linalg.matmul
// CHECK:         linalg.batch_matmul

// -----

func.func @batch_matmul_truncf_f16f16f32(%arg0 : tensor<64x100x250xf32>, %arg1 : tensor<64x250x500xf32>,
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
  return %1 : tensor<64x100x500xf32>
}

//      CHECK: func @batch_matmul_truncf_f16f16f32(%[[ARG0:.+]]: tensor<64x100x250xf32>, %[[ARG1:.+]]: tensor<64x250x500xf32>
//  CHECK-DAG: %[[INIT:.+]] = tensor.empty() : tensor<64x250x500xf16>
//  CHECK-DAG: arith.truncf %[[ARG0]] : tensor<64x100x250xf32> to tensor<64x100x250xf16>
//      CHECK: linalg.generic
// CHECK-SAME:   ins(%[[ARG1]] : tensor<64x250x500xf32>)
// CHECK-SAME:   outs(%[[INIT]] : tensor<64x250x500xf16>)
//      CHECK: element_types = [f16, f16, f32]

// -----

func.func @batch_matmul_casted_ui8i8i32(%arg0 : tensor<64x100x250xi8>, %arg1 : tensor<64x250x500xi8>,
      %arg2 : tensor<64x100x500xi32>) -> tensor<64x100x500xi32> {
  %0 = tensor.empty() : tensor<64x250x500xi32>
  %casted0 = arith.extui %arg0 : tensor<64x100x250xi8> to tensor<64x100x250xi32>
  %casted1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                              iterator_types = ["parallel", "parallel", "parallel"]}
                              ins(%arg1 : tensor<64x250x500xi8>)
                              outs(%0 : tensor<64x250x500xi32>) {
  ^bb0(%in: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      linalg.yield %2 : i32
  } -> tensor<64x250x500xi32>
  %1 = linalg.batch_matmul ins(%casted0, %casted1 : tensor<64x100x250xi32>, tensor<64x250x500xi32>)
      outs(%arg2 : tensor<64x100x500xi32>) -> tensor<64x100x500xi32>
  return %1 : tensor<64x100x500xi32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0 + 250)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0 + 100)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0 + 64)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0 + 500)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @batch_matmul_casted_ui8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xi32>
//  CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:     %[[C100:.+]] = arith.constant 100 : index
//  CHECK-DAG:     %[[C250:.+]] = arith.constant 250 : index
//  CHECK-DAG:     %[[C500:.+]] = arith.constant 500 : index
//      CHECK:   %[[LHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x250xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [ui8, i8, i32]>> -> index, index, index
//      CHECK:   %[[LHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[LHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[%[[LHS_PADDING_SIZE0]], %[[LHS_PADDING_SIZE1]], %[[LHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x250xi8> to tensor<?x?x?xi8>
//  CHECK-DAG:   %[[LHS_DIM0:.+]] = affine.apply #[[MAP1]]()[%[[LHS_TILE_SIZE]]#2, %[[C250]]]
//  CHECK-DAG:   %[[LHS_DIM1:.+]] = affine.apply #[[MAP2]]()[%[[LHS_TILE_SIZE]]#1, %[[C100]]]
//  CHECK-DAG:   %[[LHS_DIM2:.+]] = affine.apply #[[MAP3]]()[%[[LHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [ui8, i8, i32], original_type = tensor<64x100x250xi8>>>
//      CHECK:   %[[INIT_LHS_CAST:.+]] = tensor.empty(%[[LHS_DIM2]], %[[LHS_DIM1]], %[[LHS_DIM0]]) : tensor<?x?x?xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [ui8, i8, i32], original_type = tensor<64x100x250xi8>>>
//      CHECK:   %[[LHS_CASTED:.+]] = linalg.generic {indexing_maps = [#[[MAP5]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:       ins(%[[LHS]] : tensor<?x?x?xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [ui8, i8, i32], original_type = tensor<64x100x250xi8>>>)
// CHECK-SAME:       outs(%[[INIT_LHS_CAST]] : tensor<?x?x?xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [ui8, i8, i32], original_type = tensor<64x100x250xi8>>>)
// CHECK-NEXT:   ^bb0(%[[LHS_ARG_IN:.+]]: i8, %[[LHS_ARG_OUT:.+]]: i32):
// CHECK-NEXT:   %[[LHS_CAST_OP:.+]] = arith.extui %[[LHS_ARG_IN]] : i8 to i32
// CHECK-NEXT:   linalg.yield %[[LHS_CAST_OP]] : i32
// CHECK-NEXT:   -> tensor<?x?x?xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [ui8, i8, i32], original_type = tensor<64x100x250xi8>>>
//      CHECK:   %[[RHS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x250x500xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [ui8, i8, i32]>> -> index, index, index
//      CHECK:   %[[RHS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//      CHECK:   %[[RHS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[%[[RHS_PADDING_SIZE0]], %[[RHS_PADDING_SIZE1]], %[[RHS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x250x500xi8> to tensor<?x?x?xi8>
//  CHECK-DAG:   %[[RHS_DIM0:.+]] = affine.apply #[[MAP4]]()[%[[RHS_TILE_SIZE]]#2, %[[C500]]]
//  CHECK-DAG:   %[[RHS_DIM1:.+]] = affine.apply #[[MAP1]]()[%[[RHS_TILE_SIZE]]#1, %[[C250]]]
//  CHECK-DAG:   %[[RHS_DIM2:.+]] = affine.apply #[[MAP3]]()[%[[RHS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [ui8, i8, i32], original_type = tensor<64x250x500xi8>>>
//      CHECK:   %[[INIT_RHS_CAST:.+]] = tensor.empty(%[[RHS_DIM2]], %[[RHS_DIM1]], %[[RHS_DIM0]]) : tensor<?x?x?xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [ui8, i8, i32], original_type = tensor<64x250x500xi8>>>
//      CHECK:   %[[RHS_CASTED:.+]] = linalg.generic {indexing_maps = [#[[MAP5]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:       ins(%[[RHS]] : tensor<?x?x?xi8, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [ui8, i8, i32], original_type = tensor<64x250x500xi8>>>)
// CHECK-SAME:       outs(%[[INIT_RHS_CAST]] : tensor<?x?x?xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [ui8, i8, i32], original_type = tensor<64x250x500xi8>>>)
// CHECK-NEXT:   ^bb0(%[[RHS_ARG_IN:.+]]: i8, %[[RHS_ARG_OUT:.+]]: i32):
// CHECK-NEXT:   %[[RHS_CAST_OP:.+]] = arith.extsi %[[RHS_ARG_IN]] : i8 to i32
// CHECK-NEXT:   linalg.yield %[[RHS_CAST_OP]] : i32
// CHECK-NEXT:   -> tensor<?x?x?xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [ui8, i8, i32], original_type = tensor<64x250x500xi8>>>
//      CHECK:   %[[OUTS_TILE_SIZE:.+]]:3 = iree_linalg_ext.upper_bound_tile_size tensor<64x100x500xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [ui8, i8, i32]>> -> index, index, index
//      CHECK:   %[[OUTS_PADDING_SIZE0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#0, %[[C64]]]
//      CHECK:   %[[OUTS_PADDING_SIZE1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#1, %[[C100]]]
//      CHECK:   %[[OUTS_PADDING_SIZE2:.+]] = affine.apply #[[MAP]]()[%[[OUTS_TILE_SIZE]]#2, %[[C500]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[%[[OUTS_PADDING_SIZE0]], %[[OUTS_PADDING_SIZE1]], %[[OUTS_PADDING_SIZE2]]]
//      CHECK:       tensor<64x100x500xi32> to tensor<?x?x?xi32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?x?xi32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [ui8, i8, i32], original_type = tensor<64x100x500xi32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS_CASTED]], %[[RHS_CASTED]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0, 0] [64, 100, 500] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_casted_ui8i8i32(%arg0 : tensor<100x250xi8>, %arg1 : tensor<250x500xi8>,
      %arg2 : tensor<100x500xi32>) -> tensor<100x500xi32> {
  %0 = tensor.empty() : tensor<250x500xi32>
  %casted0 = arith.extui %arg0 : tensor<100x250xi8> to tensor<100x250xi32>
  %casted1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                              affine_map<(d0, d1) -> (d0, d1)>],
                              iterator_types = ["parallel", "parallel"]}
                              ins(%arg1 : tensor<250x500xi8>)
                              outs(%0 : tensor<250x500xi32>) {
  ^bb0(%in: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      linalg.yield %2 : i32
  } -> tensor<250x500xi32>
  %1 = linalg.matmul ins(%casted0, %casted1 : tensor<100x250xi32>, tensor<250x500xi32>)
      outs(%arg2 : tensor<100x500xi32>) -> tensor<100x500xi32>
  return %1 : tensor<100x500xi32>
}

//      CHECK: func @matmul_casted_ui8i8i32(
//      CHECK: element_types = [ui8

// -----

func.func @matmul_casted_from_i1_f32f32f32(%arg0 : tensor<64x256xi1>,
    %arg1 : tensor<256x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %casted = arith.uitofp %arg0 : tensor<64x256xi1> to tensor<64x256xf32>
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = linalg.matmul ins(%casted, %arg1 : tensor<64x256xf32>, tensor<256x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %2 : tensor<64x128xf32>
}
// CHECK-LABEL: func.func @matmul_casted_from_i1_f32f32f32
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  LHS, element_types = [f32, f32, f32], original_type = tensor<64x256xf32>>>
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  RHS, element_types = [f32, f32, f32], original_type = tensor<256x128xf32>>>
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<64x128xf32>>>

// -----

func.func @matmul_generic_casted_from_i1_f32f32f32(%arg0 : tensor<64x256xi1>,
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
  return %2 : tensor<64x128xf32>
}
// CHECK-LABEL: func.func @matmul_generic_casted_from_i1_f32f32f32
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  LHS, element_types = [f32, f32, f32], original_type = tensor<64x256xf32>>>
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  RHS, element_types = [f32, f32, f32], original_type = tensor<256x128xf32>>>
// CHECK:         set_encoding {{.+}} tensor<?x?xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<64x128xf32>>>

// -----

func.func @matmul_f32f32f32_narrow_M(%arg0 : tensor<2x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<2x500xf32>) -> tensor<2x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<2x500xf32>) -> tensor<2x500xf32>
  return %0 : tensor<2x500xf32>
}

//      CHECK: func @matmul_f32f32f32_narrow_M(
//      CHECK:  iree_linalg_ext.upper_bound_tile_size tensor<2x250xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  LHS, element_types = [f32, f32, f32], matmul_narrow_M = 2 : index>>
//      CHECK:  iree_linalg_ext.upper_bound_tile_size tensor<250x500xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  RHS, element_types = [f32, f32, f32], matmul_narrow_M = 2 : index>>
//      CHECK:  iree_linalg_ext.upper_bound_tile_size tensor<2x500xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 2 : index>>
//      CHECK:   linalg.matmul

// -----

func.func @batch_matmul_f32f32f32_narrow_MN(%arg0 : tensor<64x4x250xf32>, %arg1 : tensor<64x250x2xf32>,
    %arg2 : tensor<64x4x2xf32>) -> tensor<64x4x2xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x4x250xf32>, tensor<64x250x2xf32>)
      outs(%arg2 : tensor<64x4x2xf32>) -> tensor<64x4x2xf32>
  return %0 : tensor<64x4x2xf32>
}

//      CHECK: func @batch_matmul_f32f32f32_narrow_MN(
//      CHECK:   iree_linalg_ext.upper_bound_tile_size tensor<64x4x250xf32, #iree_linalg_ext.encoding<user =  BATCH_MATMUL, role =  LHS, element_types = [f32, f32, f32], matmul_narrow_M = 4 : index, matmul_narrow_N = 2 : index>>
//      CHECK:   iree_linalg_ext.upper_bound_tile_size tensor<64x250x2xf32, #iree_linalg_ext.encoding<user =  BATCH_MATMUL, role =  RHS, element_types = [f32, f32, f32], matmul_narrow_M = 4 : index, matmul_narrow_N = 2 : index>>
//      CHECK:   iree_linalg_ext.upper_bound_tile_size tensor<64x4x2xf32, #iree_linalg_ext.encoding<user =  BATCH_MATMUL, role =  RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 4 : index, matmul_narrow_N = 2 : index>>
//      CHECK:   linalg.batch_matmul
