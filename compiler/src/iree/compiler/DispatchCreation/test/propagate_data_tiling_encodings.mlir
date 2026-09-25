// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-propagate-data-tiling-encodings))" %s | FileCheck %s

// The same encoding rules as hoist_encoding_ops, applied before dispatch formation.
// The late pass must keep its original placement policy on these inputs.
// RUN: iree-opt --split-input-file --iree-dispatch-creation-hoist-encoding-ops %s | FileCheck %s --check-prefix=LATE

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4]>
// CHECK-LABEL: @bubble_through_dequant(
// CHECK: %[[A:.*]] = iree_encoding.set_encoding %arg0
// CHECK: %[[B:.*]] = iree_encoding.set_encoding %arg1
// CHECK: %[[C:.*]] = iree_encoding.set_encoding %arg2
// CHECK: linalg.generic {{.*}} ins(%[[A]], %[[B]], %[[C]]
// CHECK-SAME: tensor<2x11008x128xi8, #
// CHECK-NOT: iree_encoding.set_encoding
// CHECK: util.return
// LATE-LABEL: @bubble_through_dequant(
// LATE: %[[DEQUANT:.*]] = linalg.generic
// LATE: iree_encoding.set_encoding %[[DEQUANT]]
util.func public @bubble_through_dequant(
    %arg0: tensor<2x11008x128xi8>, %arg1: tensor<2x11008xf32>, %arg2: tensor<2x11008xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map, #map1, #map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg1, %arg2 : tensor<2x11008x128xi8>, tensor<2x11008xf32>, tensor<2x11008xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %18 = arith.extui %in : i8 to i32
      %19 = arith.uitofp %18 : i32 to f32
      %20 = arith.subf %19, %in_1 : f32
      %21 = arith.mulf %20, %in_0 : f32
      linalg.yield %21 : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    util.return %13 : tensor<2x11008x128xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
// CHECK-LABEL: @bubble_through_broadcast(
// CHECK: %[[A:.*]] = iree_encoding.set_encoding %arg0 : tensor<11008x128xf32> -> tensor<11008x128xf32, #
// CHECK: linalg.generic {{.*}} ins(%[[A]]
// CHECK-SAME: outs(%{{.*}} : tensor<2x11008x128xf32, #
// CHECK-NOT: iree_encoding.set_encoding
// CHECK: util.return
util.func public @bubble_through_broadcast(
    %arg0: tensor<11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<11008x128xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    util.return %13 : tensor<2x11008x128xf32, #encoding>
}

// -----

// Padding encodings are not data-tiling encodings of a contraction, so they
// are not bubbled.
#map = affine_map<(d0, d1) -> (d0, d1)>
#encoding = #iree_encoding.padding<[0, 64]>
// CHECK-LABEL: @no_bubble_padding_encoding(
// CHECK:         %[[DEQUANT:.+]] = linalg.generic
// CHECK-SAME:      ins(%{{.+}} : tensor<128x256xi8>)
// CHECK:         iree_encoding.set_encoding %[[DEQUANT]]
util.func public @no_bubble_padding_encoding(%arg0: tensor<128x256xi8>) -> tensor<128x256xf32, #encoding> {
  %empty = tensor.empty() : tensor<128x256xf32>
  %dequant = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi8>) outs(%empty : tensor<128x256xf32>) {
  ^bb0(%in: i8, %out: f32):
    %ext = arith.extui %in : i8 to i32
    %fp = arith.uitofp %ext : i32 to f32
    linalg.yield %fp : f32
  } -> tensor<128x256xf32>
  %encoded = iree_encoding.set_encoding %dequant : tensor<128x256xf32> -> tensor<128x256xf32, #encoding>
  util.return %encoded : tensor<128x256xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
// CHECK-LABEL: @propagate_unset_encoding_through_generic_with_scalar(
// CHECK-NOT: iree_encoding.unset_encoding
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<4096x?xf32, #{{.*}}>, f32)
// CHECK: iree_encoding.unset_encoding %[[RESULT]]
// LATE-LABEL: @propagate_unset_encoding_through_generic_with_scalar(
// LATE: %[[RAW:.*]] = iree_encoding.unset_encoding
// LATE: linalg.generic {{.*}} ins(%[[RAW]],
util.func public @propagate_unset_encoding_through_generic_with_scalar(%arg0: tensor<4096x?xf32, #encoding>, %arg1: f32, %arg2: index) -> tensor<4096x?xf32> {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<4096x?xf32, #encoding> -> tensor<4096x?xf32>{%arg2}
    %2 = tensor.empty(%arg2) : tensor<4096x?xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<4096x?xf32>, f32) outs(%2 : tensor<4096x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4096x?xf32>
    util.return %3 : tensor<4096x?xf32>
}

// -----

#map_sink = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_sink1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_sink2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_sink3 = affine_map<(d0, d1) -> (d0, d1)>
#map_sink4 = affine_map<(d0, d1) -> ()>
#encoding_sink = #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map_sink, #map_sink1, #map_sink2], iteration_sizes = [?, 4096, 4096]>
// CHECK-LABEL: @sink_unset_encoding_with_encoding_dims(
// CHECK: %[[SCALE:.*]] = iree_encoding.set_encoding %arg1 encoding_dims{%arg2}
// CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%arg0, %[[SCALE]]
// CHECK-SAME: tensor<?x4096xbf16, #
// CHECK: iree_encoding.unset_encoding %[[RESULT]] encoding_dims{%arg2}
util.func public @sink_unset_encoding_with_encoding_dims(%arg0: tensor<?x4096xf32, #encoding_sink>, %arg1: tensor<f32>, %m: index) -> tensor<?x4096xbf16> {
    %1 = iree_encoding.unset_encoding %arg0 encoding_dims{%m} : tensor<?x4096xf32, #encoding_sink> -> tensor<?x4096xf32>{%m}
    %2 = tensor.empty(%m) : tensor<?x4096xbf16>
    %3 = linalg.generic {indexing_maps = [#map_sink3, #map_sink4, #map_sink3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%2 : tensor<?x4096xbf16>) {
    ^bb0(%in: f32, %in_0: f32, %out: bf16):
      %4 = arith.mulf %in, %in_0 : f32
      %5 = arith.truncf %4 : f32 to bf16
      linalg.yield %5 : bf16
    } -> tensor<?x4096xbf16>
    util.return %3 : tensor<?x4096xbf16>
}

// -----

// The early placement policy must leave dispatch-contained encodings alone.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
// CHECK-LABEL: @no_bubble_inside_existing_dispatch(
// CHECK-NOT: iree_encoding.set_encoding
// CHECK: flow.dispatch.region
// CHECK: %[[BCAST:.*]] = linalg.generic
// CHECK: iree_encoding.set_encoding %[[BCAST]]
// CHECK: flow.return
util.func public @no_bubble_inside_existing_dispatch(
    %arg0: tensor<11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %6 = flow.dispatch.region -> (tensor<2x11008x128xf32, #encoding>) {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<11008x128xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    flow.return %13 : tensor<2x11008x128xf32, #encoding>
  }
  util.return %6 : tensor<2x11008x128xf32, #encoding>
}

// -----

// The broadcast producer is outside the scf.if that holds the set_encoding.
// Propagation stops at the region boundary even though the rewrite would be
// valid inside the region.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
// CHECK-LABEL: @no_bubble_across_region(
// CHECK-NOT:     iree_encoding.set_encoding
// CHECK:         %[[BCAST:.+]] = linalg.generic
// CHECK:         scf.if
// CHECK-NEXT:      iree_encoding.set_encoding %[[BCAST]]
util.func public @no_bubble_across_region(%arg0: tensor<11008x128xf32>, %init: tensor<2x11008x128xf32, #encoding>, %cond: i1) -> tensor<2x11008x128xf32, #encoding> {
  %empty = tensor.empty() : tensor<2x11008x128xf32>
  %bcast = linalg.generic {
      indexing_maps = [#map1, #map],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<11008x128xf32>)
      outs(%empty : tensor<2x11008x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x11008x128xf32>
  %result = scf.if %cond -> (tensor<2x11008x128xf32, #encoding>) {
    %encoded = iree_encoding.set_encoding %bcast : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    scf.yield %encoded : tensor<2x11008x128xf32, #encoding>
  } else {
    scf.yield %init : tensor<2x11008x128xf32, #encoding>
  }
  util.return %result : tensor<2x11008x128xf32, #encoding>
}

// -----

// The only consumer is inside an scf.if, so the unset_encoding outside of it is
// not sunk into the region, even though the rewrite would be valid.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
// CHECK-LABEL: @no_sink_across_region(
// CHECK:         %[[RAW:.+]] = iree_encoding.unset_encoding
// CHECK:         scf.if
// CHECK:           linalg.generic {{.*}} ins(%[[RAW]], %{{.+}} : tensor<4096x?xf32>, f32)
// CHECK-NOT:     iree_encoding.unset_encoding
// CHECK:         util.return
util.func public @no_sink_across_region(%arg0: tensor<4096x?xf32, #encoding>, %arg1: f32, %arg2: index, %cond: i1) -> tensor<4096x?xf32> {
  %raw = iree_encoding.unset_encoding %arg0 : tensor<4096x?xf32, #encoding> -> tensor<4096x?xf32>{%arg2}
  %empty = tensor.empty(%arg2) : tensor<4096x?xf32>
  %result = scf.if %cond -> (tensor<4096x?xf32>) {
    %scaled = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%raw, %arg1 : tensor<4096x?xf32>, f32) outs(%empty : tensor<4096x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %mul = arith.mulf %in, %in_0 : f32
      linalg.yield %mul : f32
    } -> tensor<4096x?xf32>
    scf.yield %scaled : tensor<4096x?xf32>
  } else {
    scf.yield %empty : tensor<4096x?xf32>
  }
  util.return %result : tensor<4096x?xf32>
}
