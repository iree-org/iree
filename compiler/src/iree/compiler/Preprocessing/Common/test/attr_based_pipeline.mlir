// RUN: iree-opt --iree-preprocessing-attr-based-pipeline --mlir-print-local-scope --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @test(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  return %matmul : tensor<10x40xf16>
}
// CHECK-LABEL: test
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[MATMUL:.+]] = linalg.matmul
//       CHECK:     flow.return %[[MATMUL]]
//       CHECK:   return %[[DISPATCH]]

// -----

module {
func.func @function1(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-generalize-linalg-matmul-experimental">} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  return %matmul : tensor<10x40xf16>
}
func.func @function2(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-pad-linalg-ops">} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  return %matmul : tensor<10x40xf16>
}
}
// CHECK-LABEL: func @function1
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//       CHECK:   return %[[GENERIC]]
// CHECK-LABEL: func @function2
//   CHECK-DAG:   %[[PAD1:.+]] = tensor.pad
//   CHECK-DAG:   %[[PAD2:.+]] = tensor.pad
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[PAD1]],
//  CHECK-SAME:       outs(%[[PAD2]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[MATMUL]]
//       CHECK:   return %[[SLICE]]

// -----

func.func @function(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-pad-linalg-ops,iree-preprocessing-generalize-linalg-matmul-experimental">} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  return %matmul : tensor<10x40xf16>
}
// CHECK-LABEL: func @function
//   CHECK-DAG:   %[[PAD1:.+]] = tensor.pad
//   CHECK-DAG:   %[[PAD2:.+]] = tensor.pad
//       CHECK:   %[[MATMUL:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[PAD1]],
//  CHECK-SAME:       outs(%[[PAD2]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[MATMUL]]
//       CHECK:   return %[[SLICE]]

// -----

// expected-remark@+1 {{expected preprocessing_pipeline attribute to be a `StringAttr` that specifies the pass pipeline to apply}}
func.func @function() attributes {
    preprocessing_pipeline = "iree-preprocessing-pad-linalg-ops"} {
  return
}
