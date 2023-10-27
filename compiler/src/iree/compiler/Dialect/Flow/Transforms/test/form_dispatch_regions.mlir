// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions))" --split-input-file %s | FileCheck %s

func.func @pack_elementwise_fusion(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %4 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %5 = linalg.generic  {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1, %arg0 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 :f32) :
      %6 = arith.addf %b0, %b1 : f32
      linalg.yield %6 : f32
  } -> tensor<?x?xf32>
  %6 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%d0]
  %7 = affine.apply affine_map<()[s0] -> (s0 ceildiv 32)>()[%d1]
  %8 = tensor.empty(%6, %7) : tensor<?x?x8x32xf32>
  // TODO(#12746) : The inner_tiles could be dynamic here. It is disabled
  // due to unrelated codegen issue.
  %9 = tensor.pack %5 padding_value(%cst : f32)
      inner_dims_pos = [0, 1] inner_tiles = [8, 32]
      into %8 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
  return %9 : tensor<?x?x8x32xf32>
}
// CHECK-LABEL: func @pack_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:         ins(%[[ARG1]], %[[ARG0]] :
//       CHECK:     %[[PACK:.+]] = tensor.pack %[[GENERIC]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   return %[[RETURN]]

// -----

func.func @pack_fusion(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%d0) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %3 = arith.addf %b0, %b1 : f32
      linalg.yield %3 : f32
  } -> tensor<?xf32>
  %4 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %5 = linalg.generic  {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1, %2 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 :f32) :
      %6 = arith.addf %b0, %b1 : f32
      linalg.yield %6 : f32
  } -> tensor<?x?xf32>
  %6 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%d0]
  %7 = affine.apply affine_map<()[s0] -> (s0 ceildiv 32)>()[%d1]
  %8 = tensor.empty(%6, %7) : tensor<?x?x8x32xf32>
  // TODO(#12746) : The inner_tiles could be dynamic here. It is disabled
  // due to unrelated codegen issue.
  %9 = tensor.pack %5 padding_value(%cst : f32)
      inner_dims_pos = [0, 1] inner_tiles = [8, 32]
      into %8 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
  return %9 : tensor<?x?x8x32xf32>
}
// CHECK-LABEL: func @pack_fusion(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[REDUCTION:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:         ins(%[[ARG0]] :
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:         ins(%[[ARG1]], %[[REDUCTION]] :
//       CHECK:     %[[PACK:.+]] = tensor.pack %[[GENERIC]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   return %[[RETURN]]

// -----

func.func @set_encoding_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%d0) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %3 = arith.addf %b0, %b1 : f32
      linalg.yield %3 : f32
  } -> tensor<?xf32>
  %4 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %5 = linalg.generic  {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1, %2 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 :f32) :
      %6 = arith.addf %b0, %b1 : f32
      linalg.yield %6 : f32
  } -> tensor<?x?xf32>
  %6 = iree_linalg_ext.set_encoding %5
      : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  return %6 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
}
// CHECK-LABEL: func @set_encoding_fusion(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[REDUCTION:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:         ins(%[[ARG0]] :
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:         ins(%[[ARG1]], %[[REDUCTION]] :
//       CHECK:     %[[PACK:.+]] = iree_linalg_ext.set_encoding %[[GENERIC]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   return %[[RETURN]]

// -----

func.func @set_encoding_pad_fusion(%arg0 : tensor<?x?xf32>,
    %arg1 : index, %arg2 : index) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[%arg1, %arg2] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = iree_linalg_ext.set_encoding %0
      : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  return %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
}
// CHECK-LABEL: func @set_encoding_pad_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[PAD:.+]] = tensor.pad %[[ARG0]]
//       CHECK:     %[[ENCODING:.+]] = iree_linalg_ext.set_encoding %[[PAD]]
//       CHECK:     flow.return %[[ENCODING]]
//       CHECK:   return %[[RETURN]]

// -----

func.func @set_encoding_pad_elementwise_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%d0) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %3 = arith.addf %b0, %b1 : f32
      linalg.yield %3 : f32
  } -> tensor<?xf32>
  %4 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %5 = linalg.generic  {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1, %2 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 :f32) :
      %6 = arith.addf %b0, %b1 : f32
      linalg.yield %6 : f32
  } -> tensor<?x?xf32>
  %6 = tensor.pad %5 low[0, 0] high[%arg2, %arg3] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %7 = iree_linalg_ext.set_encoding %6
      : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  return %7 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
}
// CHECK-LABEL: func @set_encoding_pad_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[REDUCTION:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:         ins(%[[ARG0]] :
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:         ins(%[[ARG1]], %[[REDUCTION]] :
//       CHECK:     %[[PAD:.+]] = tensor.pad %[[GENERIC]]
//       CHECK:     %[[PACK:.+]] = iree_linalg_ext.set_encoding %[[PAD]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   return %[[RETURN]]

// -----

func.func @unset_encoding_elementwise_fusion(
    %arg0: tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>,
    %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = iree_linalg_ext.unset_encoding %arg0
      : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?xf32>
  %3 = tensor.empty(%1, %2) : tensor<?x?xf32>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%3 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %5 = arith.addf %b0, %b1 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: func @unset_encoding_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>)
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_linalg_ext.unset_encoding %[[ARG0]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[UNSET_ENCODING]], %[[ARG1]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @unset_encoding_slice_elementwise_fusion(
    %arg0: tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>,
    %arg1: tensor<?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = iree_linalg_ext.unset_encoding %arg0
      : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, 0] [%arg2, %arg3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %2 = tensor.dim %1, %c0 : tensor<?x?xf32>
  %3 = tensor.dim %1, %c1 : tensor<?x?xf32>
  %4 = tensor.empty(%2, %3) : tensor<?x?xf32>
  %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%1, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%4 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %6 = arith.addf %b0, %b1 : f32
      linalg.yield %6 : f32
    } -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @unset_encoding_slice_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>
//       CHECK:   %[[RESULT0:.+]] = flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_linalg_ext.unset_encoding %[[ARG0]]
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[UNSET_ENCODING]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic {{.*}} ins(%[[SLICE]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   return %[[RESULT0]]

// -----

func.func @unpack_encoding_elementwise_fusion(
    %arg0: tensor<?x?x?x?xf32>,
    %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
  %folded_dim0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%d0, %d2]
  %folded_dim1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%d1, %d3]
  %dest = tensor.empty(%folded_dim0, %folded_dim1) : tensor<?x?xf32>
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [%d2, %d3]
      into %dest : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%dest : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %2 = arith.addf %b0, %b1 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @unpack_encoding_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>)
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[UNPACK]], %[[ARG1]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @data_dependent_shape(%arg0 : tensor<f32>, %arg1 : tensor<2xi32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0_i32 = tensor.extract %arg1[%c0] : tensor<2xi32>
  %d0 = arith.index_cast %d0_i32 : i32 to index
  %d1_i32 = tensor.extract %arg1[%c1] : tensor<2xi32>
  %d1 = arith.index_cast %d1_i32 : i32 to index
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %generic = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<f32>) outs(%empty : tensor<?x?xf32>) {
    ^bb0(%b0: f32, %b1 : f32):
      linalg.yield %b0 : f32
    } -> tensor<?x?xf32>
  return %generic : tensor<?x?xf32>
}
//      CHECK: func @data_dependent_shape(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<f32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<2xi32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[D0_I32:.+]] = tensor.extract %[[ARG1]][%[[C0]]]
//      CHECK:   %[[D0:.+]] = arith.index_cast %[[D0_I32]]
//      CHECK:   %[[D1_I32:.+]] = tensor.extract %[[ARG1]][%[[C1]]]
//      CHECK:   %[[D1:.+]] = arith.index_cast %[[D1_I32]]
//      CHECK:   %[[WL0:.+]] = affine.apply
// CHECK-SAME:       %[[D0]]
//      CHECK:   %[[WL1:.+]] = affine.apply
// CHECK-SAME:       %[[D1]]
//      CHECK:   flow.dispatch.region[%[[WL0]], %[[WL1]]]
//      CHECK:     count(%[[B0:.+]]: index, %[[B1:.+]]: index)
//      CHECK:       %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = flow.dispatch.workgroup_count_from_dag_root %[[B0]], %[[B1]]
//      CHECK:       flow.return %[[X]], %[[Y]], %[[Z]]

// -----

func.func @no_yield_dead_results(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg1, %arg2 : tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %1 = arith.addf %b0, %b1 : f32
      %2 = arith.addf %b0, %b2 : f32
      linalg.yield %1, %2 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
  return %0#1 : tensor<?xf32>
}
// CHECK: func @no_yield_dead_results
// CHECK:   %[[RESULT:.+]] = flow.dispatch.region 
// CHECK:     %[[GENERIC:.+]]:2 = linalg.generic
// CHECK:     flow.return %[[GENERIC]]#1
// CHECK:   return %[[RESULT]]

// -----

func.func @scf_nested_dispatch(%arg0 : tensor<?xi32>) -> (tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xi32>
  %empty = tensor.empty(%dim) : tensor<?xi32>
  %cmp = arith.cmpi eq, %dim, %c1 : index
  %scf = scf.if %cmp -> (tensor<?xi32>) {
    %new = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xi32>) outs(%empty : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %add = arith.addi %in, %in : i32
      linalg.yield %add : i32
    } -> tensor<?xi32>
    scf.yield %new : tensor<?xi32>
  } else {
    scf.yield %arg0 : tensor<?xi32>
  }

  return %scf : tensor<?xi32>
}

// CHECK-LABEL: @scf_nested_dispatch
// CHECK: scf.if
// CHECK: flow.dispatch.region
// CHECK: linalg.generic
// CHECK: scf.yield
// CHECK: scf.yield

// -----

func.func @grouped_quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1x4096xf32>
  %1 = tensor.empty() : tensor<4096x32x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %5 = arith.extui %in : i8 to i32
    %6 = arith.uitofp %5 : i32 to f32
    %7 = arith.subf %6, %in_1 : f32
    %8 = arith.mulf %7, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<4096x32x128xf32>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, 
                        affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, 
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], 
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} 
      ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.mulf %in, %in_0 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1x1x4096xf32>
  return %4 : tensor<1x1x4096xf32>
}
//       CHECK: func.func @grouped_quantized_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   return %[[DISP]]

// -----

func.func @nofill_grouped_quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>, %arg4: tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : tensor<4096x32x128xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %5 = arith.extui %in : i8 to i32
    %6 = arith.uitofp %5 : i32 to f32
    %7 = arith.subf %6, %in_1 : f32
    %8 = arith.mulf %7, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<4096x32x128xf32>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%arg4 : tensor<1x1x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.mulf %in, %in_0 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1x1x4096xf32>
  return %4 : tensor<1x1x4096xf32>
}
//       CHECK: func.func @nofill_grouped_quantized_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<1x1x4096xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  CHECK-SAME:       outs(%[[ARG4]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   return %[[DISP]]

// -----

func.func @grouped_quantized_matvec(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<32x128xf32>, %arg2: tensor<4096x32xf32>, %arg3: tensor<4096x32xf32>) -> tensor<4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4096xf32>
  %1 = tensor.empty() : tensor<4096x32x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%1 : tensor<4096x32x128xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %5 = arith.extui %in : i8 to i32
    %6 = arith.uitofp %5 : i32 to f32
    %7 = arith.subf %6, %in_1 : f32
    %8 = arith.mulf %7, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<4096x32x128xf32>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%arg1, %3 : tensor<32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.mulf %in, %in_0 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<4096xf32>
  return %4 : tensor<4096xf32>
}
//       CHECK: func.func @grouped_quantized_matvec
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<4096xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<4096xf32>)
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   return %[[DISP]]

// -----

func.func @ungrouped_quantized_matmul(%arg0: tensor<4096x32xi8>, %arg1: tensor<1x1x32xf32>, %arg2: tensor<4096x32xf32>, %arg3: tensor<4096x32xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1x4096xf32>
  %1 = tensor.empty() : tensor<4096x32xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg2, %arg3 : tensor<4096x32xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%1 : tensor<4096x32xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %5 = arith.extui %in : i8 to i32
    %6 = arith.uitofp %5 : i32 to f32
    %7 = arith.subf %6, %in_1 : f32
    %8 = arith.mulf %7, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<4096x32xf32>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%arg1, %3 : tensor<1x1x32xf32>, tensor<4096x32xf32>) outs(%2 : tensor<1x1x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.mulf %in, %in_0 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1x1x4096xf32>
  return %4 : tensor<1x1x4096xf32>
}

//       CHECK: func.func @ungrouped_quantized_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32xi8>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[DISP0:.+]] = flow.dispatch.region -> (tensor<4096x32xf32>)
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   flow.return %[[GEN0]] :
//       CHECK:   %[[DISP1:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[DISP0]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   return %[[DISP1]]

// -----

module {
  func.func @non_dequantization(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      linalg.yield %6 : f32
    } -> tensor<4096x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
        ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    return %4 : tensor<1x1x4096xf32>
  }
}
//       CHECK: func.func @non_dequantization
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[DISP0:.+]] = flow.dispatch.region -> (tensor<4096x32x128xf32>)
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   flow.return %[[GEN0]] :
//       CHECK:   %[[DISP1:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[DISP0]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   return %[[DISP1]]

// -----

func.func @non_dequantization(%arg0: tensor<4096x32x128xi32>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1x4096xf32>
  %1 = tensor.empty() : tensor<4096x32x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi32>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
  ^bb0(%in: i32, %in_0: f32, %in_1: f32, %out: f32):
    %6 = arith.uitofp %in : i32 to f32
    %7 = arith.subf %6, %in_1 : f32
    %8 = arith.mulf %7, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<4096x32x128xf32>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.mulf %in, %in_0 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1x1x4096xf32>
  return %4 : tensor<1x1x4096xf32>
}
//       CHECK: func.func @non_dequantization
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[DISP0:.+]] = flow.dispatch.region -> (tensor<4096x32x128xf32>)
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   flow.return %[[GEN0]] :
//       CHECK:   %[[DISP1:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[DISP0]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   return %[[DISP1]]
