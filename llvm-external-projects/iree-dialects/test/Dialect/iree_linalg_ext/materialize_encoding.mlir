// RUN: iree-dialects-opt --iree-linalg-ext-materialize-encoding -cse -split-input-file %s | FileCheck %s

func.func @pack_unpack_gemm_lhs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @pack_unpack_gemm_lhs(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUTER_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//  CHECK-DAG:   %[[OUTER_D1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
//      CHECK:   %[[PACK_DEST:.+]] = tensor.empty(%[[OUTER_D0]], %[[OUTER_D1]]) : tensor<?x?x8x4xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[PACK_DEST]]
//      CHECK:   %[[UNPACK_DEST:.+]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[PACK]] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_unpack_gemm_rhs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @pack_unpack_gemm_rhs(
//       CHECK:   tensor.pack
//  CHECK-SAME:     outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 4]
//       CHECK:   tensor.unpack %{{.+}} outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 4]

// -----

func.func @pack_unpack_gemm_result(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @pack_unpack_gemm_result(
//       CHECK:   tensor.pack
//  CHECK-SAME:     inner_dims_pos = [0, 1] inner_tiles = [8, 8]
//       CHECK:   tensor.unpack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [8, 8]

// -----

func.func @pack_gemm(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>, %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %pad_value = arith.constant 0.0 : f32
  %pad_lhs = tensor.pad %arg0 low[0, 0] high[4, 2] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<100x250xf32> to tensor<104x252xf32>
  %lhs = iree_linalg_ext.set_encoding %pad_lhs : tensor<104x252xf32> -> tensor<104x252xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %pad_rhs = tensor.pad %arg1 low[0, 0] high[2, 4] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<250x500xf32> to tensor<252x504xf32>
  %rhs = iree_linalg_ext.set_encoding %pad_rhs : tensor<252x504xf32> -> tensor<252x504xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %pad_output = tensor.pad %arg2 low[0, 0] high[4, 4] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<100x500xf32> to tensor<104x504xf32>
  %output = iree_linalg_ext.set_encoding %pad_output : tensor<104x504xf32> -> tensor<104x504xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %gemm_packed = linalg.matmul ins(%lhs, %rhs : tensor<104x252xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>, tensor<252x504xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>)
      outs(%output : tensor<104x504xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>) -> tensor<104x504xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %gemm = iree_linalg_ext.unset_encoding %gemm_packed : tensor<104x504xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> tensor<104x504xf32>
  %result = tensor.extract_slice %gemm[0, 0] [100, 500] [1, 1] : tensor<104x504xf32> to tensor<100x500xf32>
  return %result : tensor<100x500xf32>
}
//      CHECK: func @pack_gemm(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[CST:.+]] = arith.constant 0.0
//      CHECK:   %[[INIT_LHS:.+]] = tensor.empty() : tensor<13x63x8x4xf32>
//      CHECK:   %[[PACK_LHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG0]] padding_value(%[[CST]] : f32)
// CHECK-SAME:       into %[[INIT_LHS]]
//      CHECK:   %[[INIT_RHS:.+]] = tensor.empty() : tensor<63x63x8x4xf32>
//      CHECK:   %[[PACK_RHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG1]] padding_value(%[[CST]] : f32)
// CHECK-SAME:       into %[[INIT_RHS]]
//      CHECK:   %[[INIT_RESULT:.+]] = tensor.empty() : tensor<13x63x8x8xf32>
//      CHECK:   %[[PACK_RESULT:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG2]] padding_value(%[[CST]] : f32)
// CHECK-SAME:       into %[[INIT_RESULT]]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RESULT]] :
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_gemm_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %2 = iree_linalg_ext.set_encoding %arg2 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %3 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>)
      outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @pack_gemm_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//      CHECK:   %[[PACK_LHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG0]]
//      CHECK:   %[[PACK_RHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG1]]
//      CHECK:   %[[PACK_RESULT:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG2]]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RESULT]] :
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_gemm_fill_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>)
      outs(%3 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %5 = iree_linalg_ext.unset_encoding %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @pack_gemm_fill_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//  CHECK-DAG:   %[[OUT_D1:.+]] = affine.apply #[[MAP0]]()[%[[D1]]]
//  CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack {{.*}}%[[ARG0]]
//      CHECK:   %[[PACK_RHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG1]]
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[OUT_D0]], %[[OUT_D1]]) : tensor<?x?x8x8xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_unpack_batch_matmul_lhs(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @pack_unpack_batch_matmul_lhs(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[OUTER_D1:.+]] = affine.apply #[[MAP0]]()[%[[D1]]]
//  CHECK-DAG:   %[[OUTER_D2:.+]] = affine.apply #[[MAP1]]()[%[[D2]]]
//      CHECK:   %[[PACK_DEST:.+]] = tensor.empty(%[[D0]], %[[OUTER_D1]], %[[OUTER_D2]]) : tensor<?x?x?x8x4xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG0]] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[PACK_DEST]]
//      CHECK:   %[[UNPACK_DEST:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]]) : tensor<?x?x?xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[PACK]] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[UNPACK_DEST]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_unpack_batch_matmul_rhs(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>> -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @pack_unpack_batch_matmul_rhs(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[OUTER_D1:.+]] = affine.apply #[[MAP0]]()[%[[D2]]]
//  CHECK-DAG:   %[[OUTER_D2:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
//      CHECK:   %[[PACK_DEST:.+]] = tensor.empty(%[[D0]], %[[OUTER_D1]], %[[OUTER_D2]]) : tensor<?x?x?x8x4xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG0]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 4] into %[[PACK_DEST]]
//      CHECK:   %[[UNPACK_DEST:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]]) : tensor<?x?x?xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[PACK]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_unpack_batch_matmul_result(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @pack_unpack_batch_matmul_result(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[OUTER_D1:.+]] = affine.apply #[[MAP0]]()[%[[D1]]]
//  CHECK-DAG:   %[[OUTER_D2:.+]] = affine.apply #[[MAP0]]()[%[[D2]]]
//      CHECK:   %[[PACK_DEST:.+]] = tensor.empty(%[[D0]], %[[OUTER_D1]], %[[OUTER_D2]]) : tensor<?x?x?x8x8xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG0]] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %[[PACK_DEST]]
//      CHECK:   %[[UNPACK_DEST:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]]) : tensor<?x?x?xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[PACK]] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %[[UNPACK_DEST]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_batch_matmul(%arg0 : tensor<128x80x32xf32>, %arg1 : tensor<128x32x320xf32>, %arg2 : tensor<128x80x320xf32>) -> tensor<128x80x320xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<128x80x32xf32> -> tensor<128x80x32xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<128x32x320xf32> -> tensor<128x32x320xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %2 = iree_linalg_ext.set_encoding %arg2 : tensor<128x80x320xf32> -> tensor<128x80x320xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %3 = linalg.batch_matmul ins(%0, %1 : tensor<128x80x32xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>>, tensor<128x32x320xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>>)
      outs(%2 : tensor<128x80x320xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>) -> tensor<128x80x320xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<128x80x320xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> tensor<128x80x320xf32>
  return %4 : tensor<128x80x320xf32>
}
//      CHECK: func @pack_batch_matmul(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<128x80x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<128x32x320xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<128x80x320xf32>
//      CHECK:   %[[PACK_LHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG0]]
//      CHECK:   %[[PACK_RHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG1]]
//      CHECK:   %[[PACK_RESULT:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG2]]
//      CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RESULT]] :
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[BATCH_MMT4D]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_batch_matmul_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>, %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %2 = iree_linalg_ext.set_encoding %arg2 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %3 = linalg.batch_matmul ins(%0, %1 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>>, tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>>)
      outs(%2 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>) -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> tensor<?x?x?xf32>
  return %4 : tensor<?x?x?xf32>
}
//      CHECK: func @pack_batch_matmul_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//      CHECK:   %[[PACK_LHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG0]]
//      CHECK:   %[[PACK_RHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG1]]
//      CHECK:   %[[PACK_RESULT:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG2]]
//      CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RESULT]] :
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[BATCH_MMT4D]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_batch_matmul_fill_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %2 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>)
      -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %4 = linalg.batch_matmul ins(%0, %1 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = LHS, element_types = [f32, f32, f32]>>, tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RHS, element_types = [f32, f32, f32]>>)
      outs(%3 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>) -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %5 = iree_linalg_ext.unset_encoding %4 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<user = BATCH_MATMUL, role = RESULT, element_types = [f32, f32, f32]>> -> tensor<?x?x?xf32>
  return %5 : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @pack_batch_matmul_fill_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[OUT_D1:.+]] = affine.apply #[[MAP0]]()[%[[D1]]]
//  CHECK-DAG:   %[[OUT_D2:.+]] = affine.apply #[[MAP0]]()[%[[D2]]]
//  CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack %[[ARG0]]
//  CHECK-DAG:   %[[PACK_RHS:.+]] = tensor.pack %[[ARG1]]
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[D0]], %[[OUT_D1]], %[[OUT_D2]]) : tensor<?x?x?x8x8xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:     outs(%[[EMPTY]] : tensor<?x?x?x8x8xf32>)
//      CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[BATCH_MMT4D]]
//      CHECK:   return %[[UNPACK]]
