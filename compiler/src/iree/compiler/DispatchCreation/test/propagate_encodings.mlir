// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-propagate-encodings))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>
util.func public @propagate_encoding_through_collapse_shape(%src: tensor<2x4096x640xf16>) -> tensor<8192x640xf16, #encoding> {
  %collapsed = tensor.collapse_shape %src [[0, 1], [2]] : tensor<2x4096x640xf16> into tensor<8192x640xf16>
  %0 = iree_encoding.set_encoding %collapsed : tensor<8192x640xf16> -> tensor<8192x640xf16, #encoding>
  util.return %0 : tensor<8192x640xf16, #encoding>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]]>
// CHECK-LABEL: @propagate_encoding_through_collapse_shape(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] : tensor<2x4096x640xf16> -> tensor<2x4096x640xf16, #[[$ENCODING1]]>
// CHECK:         %[[COLLAPSE:.+]] = tensor.collapse_shape %[[SET_ENCODING]] {{\[}}[0, 1], [2]] : tensor<2x4096x640xf16, #[[$ENCODING1]]> into tensor<8192x640xf16, #[[$ENCODING0]]>
// CHECK:         util.return %[[COLLAPSE]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>
util.func public @propagate_encoding_through_collapse_shape_chain(%src: tensor<2x4096x64x10xf16>) -> tensor<8192x640xf16, #encoding> {
  %collapsed = tensor.collapse_shape %src [[0], [1], [2, 3]] : tensor<2x4096x64x10xf16> into tensor<2x4096x640xf16>
  %collapsed_0 = tensor.collapse_shape %collapsed [[0, 1], [2]] : tensor<2x4096x640xf16> into tensor<8192x640xf16>
  %0 = iree_encoding.set_encoding %collapsed_0 : tensor<8192x640xf16> -> tensor<8192x640xf16, #encoding>
  util.return %0 : tensor<8192x640xf16, #encoding>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG:   #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[MAP7:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG:   #[[MAP8:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]]>
// CHECK-DAG:   #[[$ENCODING2:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP6]], #[[MAP7]], #[[MAP8]]]>
// CHECK-LABEL: @propagate_encoding_through_collapse_shape_chain(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] : tensor<2x4096x64x10xf16> -> tensor<2x4096x64x10xf16, #[[$ENCODING1]]>
// CHECK:         %[[COLLAPSE_0:.+]] = tensor.collapse_shape %[[SET_ENCODING]] {{\[}}[0], [1], [2, 3]] : tensor<2x4096x64x10xf16, #[[$ENCODING1]]> into tensor<2x4096x640xf16, #[[$ENCODING2]]>
// CHECK:         %[[COLLAPSE_1:.+]] = tensor.collapse_shape %[[COLLAPSE_0]] {{\[}}[0, 1], [2]] : tensor<2x4096x640xf16, #[[$ENCODING2]]> into tensor<8192x640xf16, #[[$ENCODING0]]>
// CHECK:         util.return %[[COLLAPSE_1]]
