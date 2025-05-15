// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-util-attr-based-pipeline{pipeline-name=\"preprocessing\"}))" --mlir-print-local-scope --split-input-file --verify-diagnostics --iree-dispatch-creation-propagate-collapse-across-expands=true %s | FileCheck %s

module {
util.func @function1(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    util.pipelines = {preprocessing = #util.pipeline<"iree-preprocessing-generalize-linalg-matmul-experimental">}} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  util.return %matmul : tensor<10x40xf16>
}
util.func @function2(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    util.pipelines = {preprocessing = #util.pipeline<"iree-preprocessing-pad-linalg-ops">}} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  util.return %matmul : tensor<10x40xf16>
}
}
// CHECK-LABEL: @function1
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//       CHECK:   return %[[GENERIC]]
// CHECK-LABEL: @function2
//   CHECK-DAG:   %[[PAD1:.+]] = tensor.pad
//   CHECK-DAG:   %[[PAD2:.+]] = tensor.pad
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[PAD1]],
//  CHECK-SAME:       outs(%[[PAD2]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[MATMUL]]
//       CHECK:   return %[[SLICE]]

// -----

util.func @function(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    util.pipelines = {preprocessing= #util.pipeline<"iree-preprocessing-pad-linalg-ops,iree-preprocessing-generalize-linalg-matmul-experimental">}} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  util.return %matmul : tensor<10x40xf16>
}
// CHECK-LABEL: @function
//   CHECK-DAG:   %[[PAD1:.+]] = tensor.pad
//   CHECK-DAG:   %[[PAD2:.+]] = tensor.pad
//       CHECK:   %[[MATMUL:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[PAD1]],
//  CHECK-SAME:       outs(%[[PAD2]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[MATMUL]]
//       CHECK:   return %[[SLICE]]

// -----

// expected-remark@+1 {{expected preprocessing_pipeline attribute to be a `StringAttr` that specifies the pass pipeline to apply}}
util.func @function() attributes {
    util.pipelines = {preprocessing = "iree-preprocessing-pad-linalg-ops"}} {
  util.return
}
