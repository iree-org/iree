// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-propagate-encodings))" --split-input-file %s | FileCheck %s

#encoding = #iree_encoding.layout<[#iree_gpu.gpu_encoding_resolver<>]>
util.func @propagate_encoding_through_tensor_cast(%src: tensor<1024x?xf16>) -> tensor<?x512xf16, #encoding> {
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  %0 = iree_encoding.set_encoding %cast : tensor<?x512xf16> -> tensor<?x512xf16, #encoding>
  util.return %0 : tensor<?x512xf16, #encoding>
}

// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @propagate_encoding_through_tensor_cast(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] : tensor<1024x?xf16> -> tensor<1024x?xf16, #[[$ENCODING]]>
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SET_ENCODING]] : tensor<1024x?xf16, #[[$ENCODING]]> to tensor<?x512xf16, #[[$ENCODING]]>
// CHECK:         util.return %[[CAST]]
