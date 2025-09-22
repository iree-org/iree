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
  %9 = linalg.pack %5 padding_value(%cst : f32)
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
//       CHECK:     %[[PACK:.+]] = linalg.pack %[[GENERIC]]
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
  %9 = linalg.pack %5 padding_value(%cst : f32)
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
//       CHECK:     %[[PACK:.+]] = linalg.pack %[[GENERIC]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   util.return %[[RETURN]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<()[s0] -> (s0 ceildiv 8)>
#map3 = affine_map<()[s0] -> (s0 ceildiv 32)>
util.func public @transpose_pack_fusion(%arg0: tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
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
  %pack = linalg.pack %1 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %4 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
  util.return %pack : tensor<?x?x8x32xf32>
}
// No fusion as the CPU backend currently can't handle fusion with transpose
// between ops.
// CHECK-LABEL: util.func public @transpose_pack_fusion(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   %[[DISPATCH2:.+]] = flow.dispatch.region
//       CHECK:     %[[PACK:.+]] = linalg.pack %[[DISPATCH1]]
//       CHECK:     flow.return %[[PACK]]
//       CHECK:   util.return %[[DISPATCH2]]

// -----

#encoding = #iree_encoding.testing<>
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

#encoding = #iree_encoding.testing<>
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


#encoding = #iree_encoding.testing<>
util.func public @set_encoding_op(%arg0 : tensor<?x?xf32>)
    -> tensor<?x?xf32, #encoding> {
  %encode = iree_encoding.set_encoding %arg0
      : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  util.return %encode : tensor<?x?xf32, #encoding>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.testing
//      CHECK: util.func public @set_encoding_op
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//  CHECK-NOT:   flow.dispatch.region
//      CHECK:   %[[ENCODE:.+]] = iree_encoding.set_encoding %[[ARG0]]  : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>
// CHECK: util.return %[[ENCODE]]

// -----

#encoding = #iree_encoding.testing<>
util.func public @unset_encoding_elementwise_fusion(
    %arg0: tensor<?x?xf32, #encoding>,
    %arg1: tensor<?xf32>,
    %d0: index,
    %d1: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = iree_encoding.unset_encoding %arg0
      : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%d0, %d1}
  %1 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%1 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %3 = arith.addf %b0, %b1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  util.return %2 : tensor<?x?xf32>
}
//       CHECK: #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: util.func public @unset_encoding_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32, #[[$ENCODING]]>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>
//       CHECK:   %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[ARG0]]
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[UNSET_ENCODING]], %[[ARG1]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[RESULT]]

// -----

#encoding = #iree_encoding.testing<>
util.func public @unset_encoding_elementwise_fusion(
    %arg0: tensor<?x?xf32, #encoding>,
    %arg1: tensor<?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = iree_encoding.unset_encoding %arg0
      : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%arg2, %arg3}
  %1 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%1 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %3 = arith.addf %b0, %b1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  util.return %2 : tensor<?x?xf32>
}
//       CHECK: #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: util.func public @unset_encoding_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32, #[[$ENCODING]]>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>
//       CHECK:   %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[ARG0]]
//       CHECK:   %[[RESULT0:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic {{.*}} ins(%[[UNSET_ENCODING]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[RESULT0]]

// -----

util.func public @unpack_elementwise_fusion(
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
  %0 = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [%d2, %d3]
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
// CHECK-LABEL: util.func public @unpack_elementwise_fusion(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>)
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[UNPACK]], %[[ARG1]]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[RESULT]]

// -----

#encoding = #iree_encoding.testing<>
util.func public @unset_encoding_slice(%arg0: tensor<1x50x384xf32, #encoding>) -> tensor<384xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<1x50x384xf32, #encoding> -> tensor<1x50x384xf32>
  %extracted_slice = tensor.extract_slice %0[0, 0, 0] [1, 1, 384] [1, 1, 1] : tensor<1x50x384xf32> to tensor<384xf32>
  util.return %extracted_slice : tensor<384xf32>
}
// CHECK-LABEL: util.func public @unset_encoding_slice
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[UNSET_ENCODING]]
// CHECK:         util.return %[[SLICE]]

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
  %0 = linalg.unpack %arg0 inner_dims_pos = [1] inner_tiles = [%d2]
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
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
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
//      CHECK:       %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = iree_tensor_ext.dispatch.workgroup_count_from_dag_root(%[[B0]], %[[B1]])
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
  %op = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%dequant, %rhs : tensor<?x?x?xi32>, tensor<?x?x?xi32>)
      outs(%init : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  util.return %op : tensor<?x?x?xi32>
}
// CHECK-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func public @broadcasting_dequant_op(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi8>
//   CHECK-NOT:   flow.dispatch.region
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] :
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[MATMUL:.+]] = linalg.batch_matmul
//  CHECK-SAME:         indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
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

// -----

// Do not form seperate dispatches for mask generators for attention. These
// will clone into the dispatch.

util.func @attention_clone_mask(%Q : tensor<?x?xf16>, %K : tensor<?x?xf16>, %V: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %Q, %c0 : tensor<?x?xf16>
  %K2 = tensor.dim %K, %c1 : tensor<?x?xf16>
  %N = tensor.dim %V, %c1 : tensor<?x?xf16>

  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1

  %scale = arith.constant 1.0 : f16

  %mask_e = tensor.empty(%M, %K2) : tensor<?x?xi1>
  %out_e = tensor.empty(%M, %N) : tensor<?x?xf16>

  %causalmask = linalg.generic {indexing_maps = [affine_map<(M, K2) -> (M, K2)>], iterator_types = ["parallel", "parallel"]} outs(%mask_e : tensor<?x?xi1>) {
  ^bb0(%out: i1):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %dec = arith.cmpi sge, %i, %j : index
    %mask = arith.select %dec, %false, %true : i1
    linalg.yield %mask : i1
  } -> tensor<?x?xi1>

  %out = iree_linalg_ext.attention {
    indexing_maps = [
      affine_map<(M, N, K2, K1) -> (M, K1)>,
      affine_map<(M, N, K2, K1) -> (K2, K1)>,
      affine_map<(M, N, K2, K1) -> (K2, N)>,
      affine_map<(M, N, K2, K1) -> ()>,
      affine_map<(M, N, K2, K1) -> (K2, K1)>,
      affine_map<(M, N, K2, K1) -> (M, N)>
    ]
  } ins(%Q, %K, %V, %scale, %causalmask : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, f16, tensor<?x?xi1>)
  outs(%out_e : tensor<?x?xf16>) {
  ^bb0(%score : f32):
      iree_linalg_ext.yield %score : f32
  }-> tensor<?x?xf16>

  util.return %out : tensor<?x?xf16>
}

// CHECK-LABEL: @attention_clone_mask
// CHECK-NOT: flow.dispatch.region
// CHECK:     linalg.generic
// CHECK:     flow.dispatch.region
// CHECK:       iree_linalg_ext.attention
// CHECK:       flow.return

// -----

util.func @scatter_no_index_producer_fusion(%arg0 : tensor<?x1xi64>,
    %arg1 : index, %arg2 : tensor<?x1x32x8x128xf16>,
    %arg3 : tensor<?x32x8x128xf16>) -> tensor<?x32x8x128xf16> {
  %empty = tensor.empty(%arg1) : tensor<?x1xi32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x1xi64>) outs(%empty : tensor<?x1xi32>) {
  ^bb0(%in: i64, %out: i32):
    %1 = arith.trunci %in : i64 to i32
    linalg.yield %1 : i32
  } -> tensor<?x1xi32>
  %1 = iree_linalg_ext.scatter
      dimension_map = [0] unique_indices(true)
      ins(%arg2, %0 : tensor<?x1x32x8x128xf16>, tensor<?x1xi32>)
      outs(%arg3 : tensor<?x32x8x128xf16>) {
  ^bb0(%arg6: f16, %arg7: f16):
    iree_linalg_ext.yield %arg6 : f16
  } -> tensor<?x32x8x128xf16>
  util.return %1 : tensor<?x32x8x128xf16>
}
// Indices operand should be cloned.
//
// CHECK-LABEL: func public @scatter_no_index_producer_fusion
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:         ins(%{{.+}}, %[[GENERIC]] :
//       CHECK:     flow.return %[[SCATTER]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func @move_captured_from_above_ops(%arg0 : tensor<1x1x2x4xi32>,
    %arg1 : f64, %arg2 : f64) -> tensor<2x3xi8> {
  %empty = tensor.empty() : tensor<2x3xi32>
  %unpack = linalg.unpack %arg0 outer_dims_perm = [0, 1]
      inner_dims_pos = [0, 1] inner_tiles = [2, 4] into %empty : tensor<1x1x2x4xi32> -> tensor<2x3xi32>
  %0 = arith.mulf %arg1, %arg2 : f64
  %1 = tensor.empty() : tensor<2x3xi8>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%unpack : tensor<2x3xi32>) outs(%1 : tensor<2x3xi8>) {
  ^bb0(%in: i32, %out: i8):
    %3 = arith.sitofp %in : i32 to f32
    %4 = arith.truncf %0 : f64 to f32
    %5 = arith.mulf %3, %4 : f32
    %48 = arith.fptosi %5 : f32 to i8
    linalg.yield %48 : i8
  } -> tensor<2x3xi8>
  util.return %2 : tensor<2x3xi8>
}
// CHECK-LABEL: func public @move_captured_from_above_ops
//       CHECK:   %[[OP:.+]] = arith.mulf
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[UNPACK]] :
//       CHECK:       %[[TRUNCF:.+]] = arith.truncf %[[OP]]
//       CHECK:       linalg.yield
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func @horizontal_fusion1(%lhs : tensor<2x4096x640xf16>,
    %rhs0 : tensor<10x64x640xf16>, %rhs1 : tensor<10x64x640xf16>,
    %rhs2 : tensor<10x64x640xf16>) ->
    (tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>,
     tensor<2x10x4096x64xf16>) {
  %4 = tensor.empty() : tensor<2x10x4096x64xf32>
  %cst = arith.constant 0.0 : f32
  %5 = linalg.fill ins(%cst : f32)
      outs(%4 : tensor<2x10x4096x64xf32>) -> tensor<2x10x4096x64xf32>
  %6:3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs0, %rhs1, %rhs2
          : tensor<2x4096x640xf16>, tensor<10x64x640xf16>, tensor<10x64x640xf16>,
            tensor<10x64x640xf16>)
      outs(%5, %5, %5
          : tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>) {
  ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f32, %out_3: f32, %out_4: f32):
    %14 = arith.extf %in : f16 to f32
    %15 = arith.extf %in_0 : f16 to f32
    %16 = arith.mulf %14, %15 : f32
    %17 = arith.addf %out, %16 : f32
    %18 = arith.extf %in_1 : f16 to f32
    %19 = arith.mulf %14, %18 : f32
    %20 = arith.addf %out_3, %19 : f32
    %21 = arith.extf %in_2 : f16 to f32
    %22 = arith.mulf %14, %21 : f32
    %23 = arith.addf %out_4, %22 : f32
    linalg.yield %17, %20, %23 : f32, f32, f32
  } -> (tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>)
  %7 = tensor.empty() : tensor<2x10x4096x64xf16>
  %8 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%6#0 : tensor<2x10x4096x64xf32>) outs(%7 : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %14 = arith.truncf %in : f32 to f16
    linalg.yield %14 : f16
  } -> tensor<2x10x4096x64xf16>
  %9 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%6#1 : tensor<2x10x4096x64xf32>) outs(%7 : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %14 = arith.truncf %in : f32 to f16
    linalg.yield %14 : f16
  } -> tensor<2x10x4096x64xf16>
  %10 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%6#2 : tensor<2x10x4096x64xf32>) outs(%7 : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %14 = arith.truncf %in : f32 to f16
    linalg.yield %14 : f16
  } -> tensor<2x10x4096x64xf16>
  util.return %8, %9, %10 : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>
}
// CHECK-LABEL: func public @horizontal_fusion1
//       CHECK:   %[[DISPATCH:.+]]:3 = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]]:3 = linalg.generic
//       CHECK:     %[[TRUNC0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC]]#0 :
//       CHECK:     %[[TRUNC1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC]]#1 :
//       CHECK:     %[[TRUNC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC]]#2 :
//       CHECK:     flow.return %[[TRUNC0]], %[[TRUNC1]], %[[TRUNC2]]
//       CHECK:   util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1, %[[DISPATCH]]#2

// -----

util.func @horizontal_fusion2(%lhs : tensor<2x4096x640xi8>,
    %rhs0 : tensor<2x640x640xi8>, %rhs1 : tensor<2x640x640xi8>)
    -> tensor<2x4096x640xf16> {
  %c0_i32 = arith.constant 32 : i32
  %0 = tensor.empty() : tensor<2x4096x640xf16>
  %1 = tensor.empty() : tensor<2x4096x640xi32>
  %2 = linalg.fill ins(%c0_i32 : i32)
      outs(%1 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %3:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs0, %rhs1
          : tensor<2x4096x640xi8>, tensor<2x640x640xi8>, tensor<2x640x640xi8>)
      outs(%2, %2 : tensor<2x4096x640xi32>, tensor<2x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %out: i32, %out_2: i32):
      %4 = arith.extsi %in : i8 to i32
      %5 = arith.extsi %in_0 : i8 to i32
      %6 = arith.muli %4, %5 : i32
      %7 = arith.addi %out, %6 : i32
      %8 = arith.extsi %in_1 : i8 to i32
      %9 = arith.muli %7, %8 : i32
      %10 = arith.addi %out_2, %9 : i32
      linalg.yield %7, %10 : i32, i32
  } -> (tensor<2x4096x640xi32>, tensor<2x4096x640xi32>)
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%3#1, %3#0 : tensor<2x4096x640xi32>, tensor<2x4096x640xi32>)
      outs(%0 : tensor<2x4096x640xf16>) {
  ^bb0(%in: i32, %in_0: i32, %out: f16):
    %5 = arith.sitofp %in : i32 to f32
    %6 = arith.truncf %5 : f32 to f16
    %7 = arith.sitofp %in_0 : i32 to f32
    %8 = arith.truncf %7 : f32 to f16
    %9 = arith.addf %6, %8 : f16
    linalg.yield %9 : f16
  } -> tensor<2x4096x640xf16>
  util.return %4 : tensor<2x4096x640xf16>
}
// CHECK-LABEL: func public @horizontal_fusion2
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]]:2 = linalg.generic
//       CHECK:     %[[TRUNC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC]]#1, %[[GENERIC]]#0 :
//       CHECK:     flow.return %[[TRUNC]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func @avoid_use_def_violation_on_consumer_fusion(%arg0 : tensor<?xf32>,
    %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<f32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %1 = arith.addf %b0, %b1 : f32
      linalg.yield %1 : f32
    } -> tensor<f32>
  %1 = util.optimization_barrier %0 : tensor<f32>
  %2 = tensor.empty() : tensor<f32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = []}
      ins(%0, %1 : tensor<f32>, tensor<f32>) outs(%2 : tensor<f32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %4 = arith.mulf %b0, %b1 : f32
      linalg.yield %4 : f32
  } -> tensor<f32>
  util.return %3 : tensor<f32>
}
// CHECK-LABEL: func public @avoid_use_def_violation_on_consumer_fusion
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//       CHECK:     flow.return %[[GENERIC1]]
//       CHECK:   %[[BARRIER:.+]] = util.optimization_barrier %[[DISPATCH1]]
//       CHECK:   %[[DISPATCH2:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[DISPATCH1]], %[[BARRIER]] :
//       CHECK:     flow.return %[[GENERIC2]]
//       CHECK:   util.return %[[DISPATCH2]]

// -----

util.func @horizontal_fusion3(%lhs : tensor<2x4096x640xf16>,
    %rhs0 : tensor<10x64x640xf16>, %rhs1 : tensor<10x64x640xf16>,
    %rhs2 : tensor<10x64x640xf16>) ->
    (tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>,
     tensor<2x10x64x4096xf16>) {
  %0 = tensor.empty() : tensor<2x10x64x4096xf32>
  %4 = tensor.empty() : tensor<2x10x4096x64xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32)
      outs(%0 : tensor<2x10x64x4096xf32>) -> tensor<2x10x64x4096xf32>
  %5 = linalg.fill ins(%cst : f32)
      outs(%4 : tensor<2x10x4096x64xf32>) -> tensor<2x10x4096x64xf32>
  %6:3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs0, %rhs1, %rhs2
          : tensor<2x4096x640xf16>, tensor<10x64x640xf16>, tensor<10x64x640xf16>,
            tensor<10x64x640xf16>)
      outs(%5, %5, %1
          : tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x64x4096xf32>) {
  ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f32, %out_3: f32, %out_4: f32):
    %14 = arith.extf %in : f16 to f32
    %15 = arith.extf %in_0 : f16 to f32
    %16 = arith.mulf %14, %15 : f32
    %17 = arith.addf %out, %16 : f32
    %18 = arith.extf %in_1 : f16 to f32
    %19 = arith.mulf %14, %18 : f32
    %20 = arith.addf %out_3, %19 : f32
    %21 = arith.extf %in_2 : f16 to f32
    %22 = arith.mulf %14, %21 : f32
    %23 = arith.addf %out_4, %22 : f32
    linalg.yield %17, %20, %23 : f32, f32, f32
  } -> (tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x64x4096xf32>)
  %7 = tensor.empty() : tensor<2x10x4096x64xf16>
  %8 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%6#0 : tensor<2x10x4096x64xf32>) outs(%7 : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %14 = arith.truncf %in : f32 to f16
    linalg.yield %14 : f16
  } -> tensor<2x10x4096x64xf16>
  %9 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%6#1 : tensor<2x10x4096x64xf32>) outs(%7 : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %14 = arith.truncf %in : f32 to f16
    linalg.yield %14 : f16
  } -> tensor<2x10x4096x64xf16>
  %2 = tensor.empty() : tensor<2x10x64x4096xf16>
  %10 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%6#2 : tensor<2x10x64x4096xf32>) outs(%2 : tensor<2x10x64x4096xf16>) {
  ^bb0(%in: f32, %out: f16):
    %14 = arith.truncf %in : f32 to f16
    linalg.yield %14 : f16
  } -> tensor<2x10x64x4096xf16>
  util.return %8, %9, %10 : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x64x4096xf16>
}
//      CHECK: #[[INTERCHANGED_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
//      CHECK: func public @horizontal_fusion3
//      CHECK:   %[[DISPATCH:.+]]:3 = flow.dispatch.region
//      CHECK:     %[[GENERIC:.+]]:3 = linalg.generic
//      CHECK:     %[[TRUNC0:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[GENERIC]]#0 :
//      CHECK:     %[[TRUNC1:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[GENERIC]]#1 :
//      CHECK:     %[[TRUNC2:.+]] = linalg.generic
// CHECK-SANE:         indexing_maps = [#[[INTERCHANGED_MAP]], #[[INTERCHANGED_MAP]]]
// CHECK-SAME:         ins(%[[GENERIC]]#2 :
//      CHECK:     flow.return %[[TRUNC0]], %[[TRUNC1]], %[[TRUNC2]]
//      CHECK:   util.return %[[DISPATCH]]#0, %[[DISPATCH]]#1, %[[DISPATCH]]#2

// -----

// Fuse rope computation only with query and not key/value
util.func @attention_rope_fusion(%arg0: tensor<10x20x30x50xbf16>,
    %arg1: tensor<10x20x40x50xbf16>, %arg2: tensor<10x20x40x50xbf16>,
    %cst : bf16) -> tensor<10x20x30x40xbf16> {
  %query_empty = tensor.empty() : tensor<10x20x30x50xbf16>
  %query = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%query_empty : tensor<10x20x30x50xbf16>) {
    ^bb0(%b0: bf16) :
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %idx2 = linalg.index 2 : index
      %idx3 = linalg.index 3 : index
      %val = tensor.extract %arg0[%idx0, %idx1, %idx2, %idx3] : tensor<10x20x30x50xbf16>
      linalg.yield %val : bf16
  } -> tensor<10x20x30x50xbf16>
  %key_empty = tensor.empty() : tensor<10x20x40x50xbf16>
  %key = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%key_empty : tensor<10x20x40x50xbf16>) {
    ^bb0(%b0: bf16) :
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %idx2 = linalg.index 2 : index
      %idx3 = linalg.index 3 : index
      %val = tensor.extract %arg1[%idx0, %idx1, %idx2, %idx3] : tensor<10x20x40x50xbf16>
      linalg.yield %val : bf16
  } -> tensor<10x20x40x50xbf16>
  %value_empty = tensor.empty() : tensor<10x20x40x50xbf16>
  %value = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%value_empty : tensor<10x20x40x50xbf16>) {
    ^bb0(%b0: bf16) :
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %idx2 = linalg.index 2 : index
      %idx3 = linalg.index 3 : index
      %val = tensor.extract %arg2[%idx0, %idx1, %idx2, %idx3] : tensor<10x20x40x50xbf16>
      linalg.yield %val : bf16
  } -> tensor<10x20x40x50xbf16>
  %empty = tensor.empty() : tensor<10x20x30x40xbf16>
  %attention = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>]}
      ins(%query, %key, %value, %cst
          : tensor<10x20x30x50xbf16>, tensor<10x20x40x50xbf16>, tensor<10x20x40x50xbf16>, bf16)
      outs(%empty : tensor<10x20x30x40xbf16>) {
    ^bb0(%arg6: f32):
      iree_linalg_ext.yield %arg6 : f32
  } -> tensor<10x20x30x40xbf16>
  util.return %attention : tensor<10x20x30x40xbf16>
}
// CHECK-LABEL: func public @attention_rope_fusion
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20x30x50xbf16>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x20x40x50xbf16>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<10x20x40x50xbf16>
//       CHECK:   %[[Q:.+]] = linalg.generic
//       CHECK:   %[[K:.+]] = linalg.generic
//       CHECK:   %[[V:.+]] = linalg.generic
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:         ins(%[[Q]], %[[K]], %[[V]]
//       CHECK:     flow.return %[[ATTENTION]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

// Avoid fusing consumer when the producer/consumer has the following structure
//
// ```mlir
// %producer = "producer_op"
// %root = "root_op"(%producer)
// %0 = "non_fusable_op"(%producer)
// %1 = "consumer_op"(%producer, %root_op, %0)
// ```
//
// Moving the `"producer_op"`, `"root+_op"`, and  `"consumer_op"`  into a dispatch
// and leaving `"non_fusable_op"` out would lead to SSA violation.
util.func public @avoid_illegal_consumer_fusion(%arg0: tensor<75600x5120xbf16>) -> tensor<75600x1x5120xbf16> {
  %cst0 = arith.constant 0.0 : bf16
  %0 = tensor.empty() : tensor<75600x5120xbf16>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<75600x5120xbf16>) outs(%0 : tensor<75600x5120xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %13 = arith.addf %in, %in : bf16
    linalg.yield %13 : bf16
  } -> tensor<75600x5120xbf16>
  %2 = tensor.empty() : tensor<75600xbf16>
  %3 = linalg.fill ins(%cst0 : bf16) outs(%2 : tensor<75600xbf16>) -> tensor<75600xbf16>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%1 : tensor<75600x5120xbf16>) outs(%3 : tensor<75600xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %8 = arith.addf %in, %out : bf16
    linalg.yield %8 : bf16
  } -> tensor<75600xbf16>
  %expanded = tensor.expand_shape %1 [[0], [1, 2]] output_shape [75600, 1, 5120]
      : tensor<75600x5120xbf16> into tensor<75600x1x5120xbf16>
  %5 = tensor.empty() : tensor<75600x1x5120xbf16>
  %6 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%expanded, %4 : tensor<75600x1x5120xbf16>, tensor<75600xbf16>)
      outs(%5 : tensor<75600x1x5120xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %9 = arith.subf %in, %in_0 : bf16
    linalg.yield %9 : bf16
  } -> tensor<75600x1x5120xbf16>
  util.return %6 : tensor<75600x1x5120xbf16>
}
// CHECK-LABEL: @avoid_illegal_consumer_fusion(
//       CHECK:   %[[DISPATCH:.+]]:2 = flow.dispatch.region
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC0]] :
//       CHECK:     flow.return %[[GENERIC1]], %[[GENERIC0]]
//       CHECK:   %[[EXPAND_SHAPE:.+]] = tensor.expand_shape %[[DISPATCH]]#1
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[EXPAND_SHAPE]], %[[DISPATCH]]#0 :
//       CHECK:   util.return %[[GENERIC2]]

// -----

util.func @interchange_producer(%update : tensor<2x2xi32>, %indices : tensor<2x2x2xi32>, %original : tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%update : tensor<2x2xi32>)
      outs(%update : tensor<2x2xi32>){
      ^bb0(%b0 : i32, %out : i32):
        linalg.yield %b0 : i32
  } -> tensor<2x2xi32>
  %result = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
                          ins(%0, %indices : tensor<2x2xi32>, tensor<2x2x2xi32>)
                          outs(%original : tensor<2x2xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<2x2xi32>
  util.return %result : tensor<2x2xi32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func public @interchange_producer
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x2xi32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<2x2x2xi32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<2x2xi32>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[TPOS:.+]] = linalg.generic
//  CHECK-SANE:       indexing_maps = [#[[MAP1]], #[[MAP0]]]
//       CHECK:     %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[TPOS]]
//       CHECK:     flow.return %[[SCATTER]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

// Check that the null indexing map is handled without a crash.
util.func @no_interchange_producer_crash(%update : tensor<2x2xi32>, %indices : tensor<2x2x2xi32>, %original : tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%original : tensor<2x2xi32>)
      outs(%original : tensor<2x2xi32>){
      ^bb0(%b0 : i32, %out : i32):
        linalg.yield %b0 : i32
  } -> tensor<2x2xi32>
  %result = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
                          ins(%update, %indices : tensor<2x2xi32>, tensor<2x2x2xi32>)
                          outs(%0 : tensor<2x2xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<2x2xi32>
  util.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func public @no_interchange_producer_crash
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//       CHECK:     %[[TPOS:.+]] = linalg.generic
//       CHECK:     flow.return %[[TPOS]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       outs(%[[DISPATCH0]]
//       CHECK:     flow.return %[[SCATTER]]
//       CHECK:   util.return %[[DISPATCH1]]

// -----

util.func public @place_truncf_in_producer_dispatch(%arg0: tensor<75600x5120xf32>, %arg1: tensor<1xi64>, %arg2: tensor<100x75600xbf16>) -> tensor<100x75600xbf16> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<75600xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<75600xf32>) -> tensor<75600xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<75600x5120xf32>) outs(%1 : tensor<75600xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<75600xf32>
  %3 = tensor.empty() : tensor<75600xbf16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<75600xf32>) outs(%3 : tensor<75600xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %6 = arith.truncf %in : f32 to bf16
    linalg.yield %6 : bf16
  } -> tensor<75600xbf16>
  %5 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%4, %arg1 : tensor<75600xbf16>, tensor<1xi64>) outs(%arg2 : tensor<100x75600xbf16>) {
  ^bb0(%arg3: bf16, %arg4: bf16):
    iree_linalg_ext.yield %arg3 : bf16
  } -> tensor<100x75600xbf16>
  util.return %5 : tensor<100x75600xbf16>
}
// CHECK-LABEL: func public @place_truncf_in_producer_dispatch
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//       CHECK:     %[[REDUCTION:.+]] = linalg.generic
//       CHECK:     %[[TRUNCF:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[REDUCTION]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[DISPATCH0]]
//       CHECK:     flow.return %[[SCATTER]]
//       CHECK:   util.return %[[DISPATCH1]]

// -----

// If the truncate cannot be fused with a producer make sure it gets fused
// with a consumer.
util.func public @place_truncf_in_consumer_dispatch(%arg0: tensor<75600xf32>, %arg1: tensor<1xi64>, %arg2: tensor<100x75600xbf16>) -> tensor<100x75600xbf16> {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.empty() : tensor<75600xbf16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : tensor<75600xf32>) outs(%3 : tensor<75600xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %6 = arith.truncf %in : f32 to bf16
    linalg.yield %6 : bf16
  } -> tensor<75600xbf16>
  %5 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%4, %arg1 : tensor<75600xbf16>, tensor<1xi64>) outs(%arg2 : tensor<100x75600xbf16>) {
  ^bb0(%arg3: bf16, %arg4: bf16):
    iree_linalg_ext.yield %arg3 : bf16
  } -> tensor<100x75600xbf16>
  util.return %5 : tensor<100x75600xbf16>
}
// CHECK-LABEL: func public @place_truncf_in_consumer_dispatch
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//       CHECK:     %[[TRUNCF:.+]] = linalg.generic
//       CHECK:     %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[TRUNCF]]
//       CHECK:     flow.return %[[SCATTER]]
//       CHECK:   util.return %[[DISPATCH0]]

// -----

// Check that the fp4 dynamic quantization kernel fuses to a single kernel.

util.func public @dynamic_quantization_fp4(%arg0 : tensor<?x32xf16>, %arg1 : index) -> (tensor<?x16xi8>, tensor<?xi8>) {
  %cst_0 = arith.constant 0.0 : f32
  %cst = arith.constant 4.000000e+00 : f32
  %2 = tensor.empty(%arg1) : tensor<?x32xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x32xf16>) outs(%2 : tensor<?x32xf32>) {
  ^bb0(%in: f16, %out: f32):
    %14 = arith.extf %in : f16 to f32
    linalg.yield %14 : f32
  } -> tensor<?x32xf32>
  %4 = tensor.empty(%arg1) : tensor<?xf32>
  %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<?xf32>) -> tensor<?xf32>
  %6 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%3 : tensor<?x32xf32>) outs(%5 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %14 = math.absf %in : f32
    %15 = arith.maximumf %14, %out : f32
    linalg.yield %15 : f32
  } -> tensor<?xf32>
  %7 = tensor.empty(%arg1) : tensor<?xi8>
  %8 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%6 : tensor<?xf32>) outs(%7 : tensor<?xi8>) {
  ^bb0(%in: f32, %out: i8):
    %14 = arith.divf %in, %cst : f32
    %15 = arith.truncf %14 : f32 to f8E8M0FNU
    %16 = arith.bitcast %15 : f8E8M0FNU to i8
    linalg.yield %16 : i8
  } -> tensor<?xi8>
  %9 = tensor.empty(%arg1) : tensor<?x32xf4E2M1FN>
  %10 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%3, %6 : tensor<?x32xf32>, tensor<?xf32>) outs(%9 : tensor<?x32xf4E2M1FN>) {
  ^bb0(%in: f32, %in_1: f32, %out: f4E2M1FN):
    %14 = arith.divf %in_1, %cst : f32
    %15 = arith.scaling_truncf %in, %14 : f32, f32 to f4E2M1FN
    linalg.yield %15 : f4E2M1FN
  } -> tensor<?x32xf4E2M1FN>
  %11 = iree_tensor_ext.bitcast %10 : tensor<?x32xf4E2M1FN>{%arg1} -> tensor<?x16xi8>{%arg1}
  util.return %11, %8 : tensor<?x16xi8>, tensor<?xi8>
}
// CHECK-LABEL: @dynamic_quantization_fp4
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x32xf16>
//       CHECK:   %[[EXTEND:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] :
//       CHECK:   %[[DISPATCH:.+]]:2 = flow.dispatch.region
//       CHECK:     %[[MAX:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[EXTEND]] :
//       CHECK:     %[[SCALE:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[MAX]] :
//       CHECK:     %[[QUANTIZED:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[EXTEND]], %[[MAX]] :
//       CHECK:     flow.return %[[SCALE]], %[[QUANTIZED]]
//       CHECK:   %[[BITCAST:.+]] = iree_tensor_ext.bitcast %[[DISPATCH]]#1
//       CHECK:   return %[[BITCAST]], %[[DISPATCH]]#0
