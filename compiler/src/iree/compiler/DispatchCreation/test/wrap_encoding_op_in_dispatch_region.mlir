// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-wrap-encoding-op-in-dispatch-region))" --split-input-file --mlir-print-local-scope %s | FileCheck %s

util.func @wrap_encoding_op(%arg0 : tensor<?x?xf32>)
    -> tensor<?x?xf32, #iree_encoding.testing_encoding<>> {
  %0 = iree_encoding.set_encoding %arg0 :
      tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.testing_encoding<>>
  util.return %0 : tensor<?x?xf32, #iree_encoding.testing_encoding<>>
}
// CHECK-LABEL: func public @wrap_encoding_op
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[ENCODING:.+]] = iree_encoding.set_encoding %[[ARG0]]
//       CHECK:     flow.return %[[ENCODING]]
//       CHECK:   return %[[DISPATCH]]

// -----

util.func @do_not_wrap_encoding_op(%arg0 : tensor<?x?xf32>)
    -> tensor<?x?xf32, #iree_encoding.testing_encoding<>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = flow.dispatch.region -> (tensor<?x?xf32, #iree_encoding.testing_encoding<>>{%d0, %d1}) {
    %1 = iree_encoding.set_encoding %arg0 :
        tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.testing_encoding<>>
    flow.return %1 : tensor<?x?xf32, #iree_encoding.testing_encoding<>>
  }
  util.return %0 : tensor<?x?xf32, #iree_encoding.testing_encoding<>>
}
// CHECK-LABEL: func public @do_not_wrap_encoding_op
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[ENCODING:.+]] = iree_encoding.set_encoding %[[ARG0]]
//       CHECK:     flow.return %[[ENCODING]]
//       CHECK:   return %[[DISPATCH]]
