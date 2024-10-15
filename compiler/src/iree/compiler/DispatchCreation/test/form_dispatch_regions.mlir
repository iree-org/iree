// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" --split-input-file %s | FileCheck %s

util.func public @pack_elementwise_fusion(%arg0 : tensor<?xf32>,
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
  util.return %9 : tensor<?x?x8x32xf32>
}
// CHECK-LABEL: util.func public @pack_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   tensor.dim
//       CHECK:   tensor.dim
//  CHECK-NOT:    tensor.dim
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:         ins(%[[ARG1]], %[[ARG0]] :
//       CHECK:     %[[PACK:.+]] = tensor.pack %[[GENERIC]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   util.return %[[RETURN]]

// -----

util.func public @pack_fusion(%arg0 : tensor<?x?xf32>,
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
  util.return %9 : tensor<?x?x8x32xf32>
}
// CHECK-LABEL: util.func public @pack_fusion(
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
//       CHECK:   util.return %[[RETURN]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<()[s0] -> (s0 ceildiv 8)>
#map3 = affine_map<()[s0] -> (s0 ceildiv 32)>
util.func public @tranpose_pack_fusion(%arg0: tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<?x?xf32>
  %2 = affine.apply #map2()[%dim]
  %3 = affine.apply #map3()[%dim_0]
  %4 = tensor.empty(%2, %3) : tensor<?x?x8x32xf32>
  %pack = tensor.pack %1 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %4 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
  util.return %pack : tensor<?x?x8x32xf32>
}
// No fusion as the CPU backend currently can't handle fusion with transpose
// between ops.
// CHECK-LABEL: util.func public @tranpose_pack_fusion(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   %[[DISPATCH2:.+]] = flow.dispatch.region
//       CHECK:     %[[PACK:.+]] = tensor.pack %[[DISPATCH1]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   util.return %[[DISPATCH2]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>
util.func public @set_encoding_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index) -> tensor<?x?xf32, #encoding> {
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
  %6 = iree_encoding.set_encoding %5
      : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  util.return %6 : tensor<?x?xf32, #encoding>
}
// CHECK-LABEL: util.func public @set_encoding_fusion(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[REDUCTION:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:         ins(%[[ARG0]] :
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:         ins(%[[ARG1]], %[[REDUCTION]] :
//       CHECK:     %[[PACK:.+]] = iree_encoding.set_encoding %[[GENERIC]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   util.return %[[RETURN]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>
util.func public @set_encoding_pad_fusion(%arg0 : tensor<?x?xf32>,
    %arg1 : index, %arg2 : index) -> tensor<?x?xf32, #encoding> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[%arg1, %arg2] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = iree_encoding.set_encoding %0
      : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  util.return %1 : tensor<?x?xf32, #encoding>
}
// CHECK-LABEL: util.func public @set_encoding_pad_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[PAD:.+]] = tensor.pad %[[ARG0]]
//       CHECK:     %[[ENCODING:.+]] = iree_encoding.set_encoding %[[PAD]]
//       CHECK:     flow.return %[[ENCODING]]
//       CHECK:   util.return %[[RETURN]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>
util.func public @set_encoding_pad_elementwise_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index) -> tensor<?x?xf32, #encoding> {
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
  %7 = iree_encoding.set_encoding %6
      : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  util.return %7 : tensor<?x?xf32, #encoding>
}
// CHECK-LABEL: util.func public @set_encoding_pad_elementwise_fusion(
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
//       CHECK:     %[[PACK:.+]] = iree_encoding.set_encoding %[[PAD]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   util.return %[[RETURN]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>
util.func public @unset_encoding_elementwise_fusion(
    %arg0: tensor<?x?xf32, #encoding>,
    %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = iree_encoding.unset_encoding %arg0
      : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>
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
  util.return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @unset_encoding_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>)
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[ARG0]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[UNSET_ENCODING]], %[[ARG1]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[RESULT]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>
util.func public @unset_encoding_slice_elementwise_fusion(
    %arg0: tensor<?x?xf32, #encoding>,
    %arg1: tensor<?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = iree_encoding.unset_encoding %arg0
      : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
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
  util.return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @unset_encoding_slice_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>
//       CHECK:   %[[RESULT0:.+]] = flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[ARG0]]
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[UNSET_ENCODING]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic {{.*}} ins(%[[SLICE]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[RESULT0]]

// -----

util.func public @unpack_encoding_elementwise_fusion(
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
  util.return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @unpack_encoding_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>)
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[UNPACK]], %[[ARG1]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @unpack_non_intersecting_reduction(
    %arg0: tensor<?x?x?xf32>,
    %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %folded_dim = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%d1, %d2]
  %dest0 = tensor.empty(%d0, %folded_dim) : tensor<?x?xf32>
  %dest1 = tensor.empty(%folded_dim) : tensor<?xf32>
  %0 = tensor.unpack %arg0 inner_dims_pos = [1] inner_tiles = [%d2]
      into %dest0 : tensor<?x?x?xf32> -> tensor<?x?xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%dest1 : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %2 = arith.addf %b0, %b1 : f32
      %3 = arith.addf %2, %b2 : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
  util.return %1 : tensor<?xf32>
}
// CHECK-LABEL: util.func public @unpack_non_intersecting_reduction(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>)
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[UNPACK]], %[[ARG1]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @data_dependent_shape(%arg0 : tensor<f32>, %arg1 : tensor<2xi32>)
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
  util.return %generic : tensor<?x?xf32>
}
//      CHECK: util.func public @data_dependent_shape(
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

util.func public @no_yield_dead_results(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg1, %arg2 : tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %1 = arith.addf %b0, %b1 : f32
      %2 = arith.addf %b0, %b2 : f32
      linalg.yield %1, %2 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
  util.return %0#1 : tensor<?xf32>
}
// CHECK: util.func public @no_yield_dead_results
// CHECK:   %[[RESULT:.+]] = flow.dispatch.region
// CHECK:     %[[GENERIC:.+]]:2 = linalg.generic
// CHECK:     flow.return %[[GENERIC]]#1
// CHECK:   util.return %[[RESULT]]

// -----

util.func public @scf_nested_dispatch(%arg0 : tensor<?xi32>) -> (tensor<?xi32>) {
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

  util.return %scf : tensor<?xi32>
}

// CHECK-LABEL: @scf_nested_dispatch
// CHECK: scf.if
// CHECK: flow.dispatch.region
// CHECK: linalg.generic
// CHECK: scf.yield
// CHECK: scf.yield

// -----

util.func public @no_dequantization_fusion(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32xf32>, %arg3: tensor<4096x32xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1x4096xf32>
  %1 = tensor.empty() : tensor<4096x32x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
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
  util.return %4 : tensor<1x1x4096xf32>
}
//       CHECK: util.func public @no_dequantization_fusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//   CHECK-DAG:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//   CHECK-DAG:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   util.return %[[DISP]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
util.func public @no_dequantization_like_fusion(%arg0: tensor<32x1x16x1x8xi16>, %arg1: tensor<32x344x16x32x8xi4>) -> tensor<32x1x344x1x32xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<32x1x16x1x8xi32>
  %1 = linalg.generic {indexing_maps = [#map, #map],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
                        ins(%arg0 : tensor<32x1x16x1x8xi16>) outs(%0 : tensor<32x1x16x1x8xi32>) {
  ^bb0(%in: i16, %out: i32):
    %7 = arith.extsi %in : i16 to i32
    linalg.yield %7 : i32
  } -> tensor<32x1x16x1x8xi32>
  %2 = tensor.empty() : tensor<32x344x16x32x8xi32>
  %3 = linalg.generic {indexing_maps = [#map, #map],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
                        ins(%arg1 : tensor<32x344x16x32x8xi4>) outs(%2 : tensor<32x344x16x32x8xi32>) {
  ^bb0(%in: i4, %out: i32):
    %7 = arith.extui %in : i4 to i32
    linalg.yield %7 : i32
  } -> tensor<32x344x16x32x8xi32>
  %4 = tensor.empty() : tensor<32x1x344x1x32xi32>
  %5 = linalg.fill ins(%c0_i32 : i32) outs(%4 : tensor<32x1x344x1x32xi32>) -> tensor<32x1x344x1x32xi32>
  %7 = linalg.batch_mmt4d ins(%1, %3 : tensor<32x1x16x1x8xi32>, tensor<32x344x16x32x8xi32>) outs(%5 : tensor<32x1x344x1x32xi32>) -> tensor<32x1x344x1x32xi32>
  util.return %7 : tensor<32x1x344x1x32xi32>
}
//       CHECK: util.func public @no_dequantization_like_fusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<32x1x16x1x8xi16>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32x344x16x32x8xi4>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:   %[[INIT0:.+]] = tensor.empty() : tensor<32x1x16x1x8xi32>
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//   CHECK-DAG:   %[[INIT2:.+]] = tensor.empty() : tensor<32x344x16x32x8xi32>
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG1]] :
//  CHECK-SAME:       outs(%[[INIT2]] :
//       CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<32x1x344x1x32xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<32x1x344x1x32xi32>)
//       CHECK:   %[[MMT4D:.+]] = linalg.batch_mmt4d
//  CHECK-SAME:       ins(%[[GEN0]], %[[GEN1]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[MMT4D]] :
//       CHECK:   util.return %[[DISP]]

// -----

util.func public @broadcasting_dequant_op(%arg0 : tensor<?x?xi8>,
    %rhs : tensor<?x?x?xi32>, %init : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d1 = tensor.dim %arg0, %c0 : tensor<?x?xi8>
  %d2 = tensor.dim %arg0, %c1 : tensor<?x?xi8>
  %d0 = tensor.dim %rhs, %c0 : tensor<?x?x?xi32>
  %empty = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xi32>
  %dequant = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xi8>) outs(%empty : tensor<?x?x?xi32>) {
    ^bb0(%in: i8, %out: i32):
      %12 = arith.extui %in : i8 to i32
      linalg.yield %12 : i32
    } -> tensor<?x?x?xi32>
  %op = linalg.batch_matmul_transpose_b
      ins(%dequant, %rhs : tensor<?x?x?xi32>, tensor<?x?x?xi32>)
      outs(%init : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  util.return %op : tensor<?x?x?xi32>
}
// CHECK-LABEL: func public @broadcasting_dequant_op(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi8>
//   CHECK-NOT:   flow.dispatch.region
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] :
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[MATMUL:.+]] = linalg.batch_matmul_transpose_b
//  CHECK-SAME:         ins(%[[GENERIC]],
//       CHECK:     flow.return %[[MATMUL]]
//       CHECK:   return %[[RETURN]]

// -----

util.func @softmax_like_fusion(%arg0: tensor<2x4096x640xf16>,
    %arg1: tensor<640xf16>, %arg2: tensor<640xf16>) -> tensor<2x4096x640x1xf16> {
  %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
      output_shape [2, 4096, 640, 1] : tensor<2x4096x640xf16> into tensor<2x4096x640x1xf16>
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.100000e+01 : f32
  %cst_1 = arith.constant 4.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x4096x640xf32>
  %1 = tensor.empty() : tensor<2x4096x640x1xf16>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<2x4096x640xf16>) outs(%0 : tensor<2x4096x640xf32>) {
    ^bb0(%in: f16, %out: f32):
      %9 = arith.extf %in : f16 to f32
      linalg.yield %9 : f32
  } -> tensor<2x4096x640xf32>
  %3 = tensor.empty() : tensor<2x4096xf32>
  %4 = linalg.fill ins(%cst : f32)
      outs(%3 : tensor<2x4096xf32>) -> tensor<2x4096xf32>
  %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%2 : tensor<2x4096x640xf32>) outs(%4 : tensor<2x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
  } -> tensor<2x4096xf32>
  %6 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%5 : tensor<2x4096xf32>) outs(%3 : tensor<2x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.divf %in, %cst_0 : f32
      linalg.yield %9 : f32
  } -> tensor<2x4096xf32>
  %7 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%2, %6 : tensor<2x4096x640xf32>, tensor<2x4096xf32>)
      outs(%4 : tensor<2x4096xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %9 = arith.subf %in, %in_4 : f32
      %10 = arith.mulf %9, %9 : f32
      %11 = arith.addf %10, %out : f32
      linalg.yield %11 : f32
  } -> tensor<2x4096xf32>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1]] output_shape [640, 1]
      : tensor<640xf16> into tensor<640x1xf16>
  %expanded_3 = tensor.expand_shape %arg2 [[0, 1]] output_shape [640, 1]
      : tensor<640xf16> into tensor<640x1xf16>
  %8 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded, %6, %7, %expanded_2, %expanded_3
          : tensor<2x4096x640x1xf16>, tensor<2x4096xf32>, tensor<2x4096xf32>,
            tensor<640x1xf16>, tensor<640x1xf16>)
      outs(%1 : tensor<2x4096x640x1xf16>) {
    ^bb0(%in: f16, %in_4: f32, %in_5: f32, %in_6: f16, %in_7: f16, %out: f16):
      %9 = arith.divf %in_5, %cst_0 : f32
      %10 = arith.addf %9, %cst_1 : f32
      %11 = math.rsqrt %10 : f32
      %12 = arith.extf %in : f16 to f32
      %13 = arith.subf %12, %in_4 : f32
      %14 = arith.mulf %13, %11 : f32
      %15 = arith.extf %in_6 : f16 to f32
      %16 = arith.mulf %14, %15 : f32
      %17 = arith.extf %in_7 : f16 to f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.truncf %18 : f32 to f16
      linalg.yield %19 : f16
  } -> tensor<2x4096x640x1xf16>
  util.return %8 : tensor<2x4096x640x1xf16>
}
// CHECK-LABEL: func public @softmax_like_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<2x4096x640xf16>
//       CHECK:   %[[BITEXTEND:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] :
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:         ins(%[[BITEXTEND]] :
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:         ins(%[[GENERIC1]] :
//       CHECK:     %[[GENERIC3:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:         ins(%[[BITEXTEND]], %[[GENERIC2]] :
//       CHECK:     %[[GENERIC4:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:         ins(%{{.+}}, %[[GENERIC2]], %[[GENERIC3]]
//       CHECK:     flow.return %[[GENERIC4]]
//       CHECK:   util.return %[[RESULT]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
util.func public @no_batch_mmt4d_fusion(%arg0: tensor<1x1x64x1x1xf32>,
    %arg1: tensor<1x32x64x4x1xf32>, %arg2: tensor<1x1x32x1x4xf32>)
    -> tensor<1x1x32x1x4xf32>  {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1x32x1x4xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x32x1x4xf32>) -> tensor<1x1x32x1x4xf32>
  %2 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x1x64x1x1xf32>, tensor<1x32x64x4x1xf32>)
  outs(%1 : tensor<1x1x32x1x4xf32>) -> tensor<1x1x32x1x4xf32>
  %3 = linalg.generic {
    indexing_maps =  [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%arg2, %2 : tensor<1x1x32x1x4xf32>, tensor<1x1x32x1x4xf32>)
    outs(%0 : tensor<1x1x32x1x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<1x1x32x1x4xf32>
  util.return %3 : tensor<1x1x32x1x4xf32>
}

// CHECK-LABEL: util.func public @no_batch_mmt4d_fusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1x64x1x1xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x32x64x4x1xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x1x32x1x4xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[INIT0:.+]] = tensor.empty() : tensor<1x1x32x1x4xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32)
//  CHECK-SAME:       outs(%[[INIT0]] : tensor<1x1x32x1x4xf32>)
//       CHECK:   %[[DISP0:.+]] = flow.dispatch.region -> (tensor<1x1x32x1x4xf32>)
//       CHECK:   %[[MMT4D:.+]] = linalg.batch_mmt4d
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] : tensor<1x1x64x1x1xf32>, tensor<1x32x64x4x1xf32>)
//  CHECK-SAME:       outs(%[[FILL]] : tensor<1x1x32x1x4xf32>)
//       CHECK:   flow.return %[[MMT4D]] : tensor<1x1x32x1x4xf32>
//       CHECK:   %[[DISP1:.+]] = flow.dispatch.region -> (tensor<1x1x32x1x4xf32>)
//       CHECK:   %[[GEN:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG2]], %[[DISP0]] : tensor<1x1x32x1x4xf32>, tensor<1x1x32x1x4xf32>)
//  CHECK-SAME:       outs(%[[INIT0]] : tensor<1x1x32x1x4xf32>)
//       CHECK:   flow.return %[[GEN]] : tensor<1x1x32x1x4xf32>
//       CHECK:   util.return %[[DISP1]] : tensor<1x1x32x1x4xf32>

// -----

util.func @custom_op_consumer_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : tensor<?xf32>):
      %1 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%b0 : tensor<?x?xf32>) outs(%b1 : tensor<?xf32>) {
        ^bb1(%bb0 : f32, %bb1 : f32) :
          %2 = arith.addf %bb0, %bb1 : f32
          linalg.yield %2 : f32
      } -> tensor<?xf32>
      iree_linalg_ext.yield %1 : tensor<?xf32>
  } -> tensor<?xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %4 = arith.mulf %b0, %b0 : f32
      linalg.yield %4 :f32
  } -> tensor<?xf32>
  util.return %3 : tensor<?xf32>
}
// CHECK-LABEL: func public @custom_op_consumer_fusion
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[CUSTOM_OP:.+]] = iree_linalg_ext.custom_op
//       CHECK:       linalg.generic
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[CUSTOM_OP]] :
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func @custom_op_producer_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %1 = arith.mulf %b0, %b0 : f32
      linalg.yield %1 :f32
  } -> tensor<?x?xf32>
  %2 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]}
      ins(%0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : tensor<?xf32>):
      %3 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%b0 : tensor<?x?xf32>) outs(%b1 : tensor<?xf32>) {
        ^bb1(%bb0 : f32, %bb1 : f32) :
          %4 = arith.addf %bb0, %bb1 : f32
          linalg.yield %4 : f32
      } -> tensor<?xf32>
      iree_linalg_ext.yield %3 : tensor<?xf32>
  } -> tensor<?xf32>
  util.return %2 : tensor<?xf32>
}
// CHECK-LABEL: func public @custom_op_producer_fusion
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//       CHECK:     %[[CUSTOM_OP:.+]] = iree_linalg_ext.custom_op
//  CHECK-SAME:         ins(%[[GENERIC]] :
//       CHECK:     flow.return %[[CUSTOM_OP]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func @custom_op_no_producer_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>, %arg4 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %1 = arith.mulf %b0, %b0 : f32
      linalg.yield %1 :f32
  } -> tensor<?x?xf32>
  %2 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (d0, s0)>,
                       affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                       affine_map<(d0, d1)[s0, s1] -> (d0, s1)>,
                       affine_map<(d0, d1)[s0, s1] -> (s1, d1)>,
                       affine_map<(d0, d1)[s0, s1] -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>]}
      ins(%0, %arg1, %arg2, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg4 : tensor<?x?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : tensor<?x?xf32>, %b2 : tensor<?x?xf32>, %b3 : tensor<?x?xf32>, %b4 : tensor<?x?xf32>):
      %3 = linalg.matmul ins(%b0, %b1 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%b2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %4 = linalg.matmul ins(%3, %b3 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%b4 : tensor<?x?xf32>) -> tensor<?x?xf32>
      iree_linalg_ext.yield %4 : tensor<?x?xf32>
  } -> tensor<?x?xf32>
  util.return %2 : tensor<?x?xf32>
}
// CHECK-LABEL: func public @custom_op_no_producer_fusion
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   %[[DISPATCH2:.+]] = flow.dispatch.region
//       CHECK:     %[[CUSTOM_OP:.+]] = iree_linalg_ext.custom_op
//  CHECK-SAME:         ins(%[[DISPATCH1]],
//       CHECK:     flow.return %[[CUSTOM_OP]]
//       CHECK:   util.return %[[DISPATCH2]]
