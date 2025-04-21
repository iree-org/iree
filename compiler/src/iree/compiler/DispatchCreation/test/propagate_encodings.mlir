// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-propagate-encodings))" --split-input-file %s | FileCheck %s

#encoding = #iree_encoding.matmul_k<k_dims = [1]>
util.func public @propagate_encoding_through_collapse_shape(%src: tensor<2x4096x640xf16>) -> tensor<8192x640xf16, #encoding> {
  %collapsed = tensor.collapse_shape %src [[0, 1], [2]] : tensor<2x4096x640xf16> into tensor<8192x640xf16>
  %0 = iree_encoding.set_encoding %collapsed : tensor<8192x640xf16> -> tensor<8192x640xf16, #encoding>
  util.return %0 : tensor<8192x640xf16, #encoding>
}
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.matmul_k<k_dims = [1]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.matmul_k<k_dims = [2]>
// CHECK-LABEL: @propagate_encoding_through_collapse_shape(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] : tensor<2x4096x640xf16> -> tensor<2x4096x640xf16, #[[$ENCODING1]]>
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SET_ENCODING]] {{\[}}[0, 1], [2]] : tensor<2x4096x640xf16, #[[$ENCODING1]]> into tensor<8192x640xf16, #[[$ENCODING0]]>
// CHECK:         util.return %[[COLLAPSED]]

// -----

#encoding = #iree_encoding.matmul_k<k_dims = [1]>
util.func public @propagate_encoding_through_collapse_shape_chain(%src: tensor<2x4096x64x10xf16>) -> tensor<8192x640xf16, #encoding> {
  %collapsed = tensor.collapse_shape %src [[0], [1], [2, 3]] : tensor<2x4096x64x10xf16> into tensor<2x4096x640xf16>
  %collapsed_0 = tensor.collapse_shape %collapsed [[0, 1], [2]] : tensor<2x4096x640xf16> into tensor<8192x640xf16>
  %0 = iree_encoding.set_encoding %collapsed_0 : tensor<8192x640xf16> -> tensor<8192x640xf16, #encoding>
  util.return %0 : tensor<8192x640xf16, #encoding>
}
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.matmul_k<k_dims = [1]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.matmul_k<k_dims = [2]>
// CHECK-LABEL: @propagate_encoding_through_collapse_shape_chain(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[COLLAPSED_0:.+]] = tensor.collapse_shape %[[SRC]] {{\[}}[0], [1], [2, 3]] : tensor<2x4096x64x10xf16> into tensor<2x4096x640xf16>
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[COLLAPSED_0]] : tensor<2x4096x640xf16> -> tensor<2x4096x640xf16, #[[$ENCODING1]]>
// CHECK:         %[[COLLAPSED_1:.+]] = tensor.collapse_shape %[[SET_ENCODING]] {{\[}}[0, 1], [2]] : tensor<2x4096x640xf16, #[[$ENCODING1]]> into tensor<8192x640xf16, #[[$ENCODING0]]>
// CHECK:         util.return %[[COLLAPSED_1]]
