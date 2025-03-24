// RUN: iree-opt %s --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-encoding-to-flow,canonicalize))" -split-input-file | FileCheck %s

#encoding = #iree_encoding.testing_encoding<>
util.func public @set_encoding_static(%arg0: tensor<123x456xf32>) -> tensor<123x456xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<123x456xf32> -> tensor<123x456xf32, #encoding>
  util.return %0 : tensor<123x456xf32, #encoding>
}
// CHECK-DAG:    #[[$ENC:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL:  @set_encoding_static(
// CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]
// CHECK:          %[[RES:.+]] = flow.tensor.encode %[[SRC]]
// CHECK-SAME:       tensor<123x456xf32> -> tensor<123x456xf32, #[[$ENC]]>

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @set_encoding_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  util.return %0 : tensor<?x?xf32, #encoding>
}
// CHECK-DAG:    #[[$ENC:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL:  @set_encoding_dynamic(
// CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:      %[[D0:.+]] = tensor.dim %[[SRC]], %[[C0]]
// CHECK-DAG:      %[[D1:.+]] = tensor.dim %[[SRC]], %[[C1]]
// CHECK:          %[[RES:.+]] = flow.tensor.encode %[[SRC]]
// CHECK-SAME:       tensor<?x?xf32>{%[[D0]], %[[D1]]} -> tensor<?x?xf32, #[[$ENC]]>{%[[D0]], %[[D1]]}

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @set_encoding_in_flow_region(%arg0: tensor<123x456xf32>) -> tensor<123x456xf32, #encoding> {
  %0 = flow.dispatch.region -> (tensor<123x456xf32, #encoding>) {
    %1 = iree_encoding.set_encoding %arg0 : tensor<123x456xf32> -> tensor<123x456xf32, #encoding>
    flow.return %1 : tensor<123x456xf32, #encoding>
  }
  util.return %0 : tensor<123x456xf32, #encoding>
}
// CHECK-LABEL: @set_encoding_in_flow_region(
// CHECK-NOT:     flow.tensor.encode
