// RUN: iree-opt --iree-preprocessing-attr-based-pipeline --mlir-print-local-scope --split-input-file %s | FileCheck %s

func.func @test(%lhs : tensor<10x20xf16>, %rhs0 : tensor<20x40xf16>,
    %rhs1 : tensor<40x80xf16>) -> tensor<10x80xf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %empty0 = tensor.empty() : tensor<10x40xf32>
  %fill0 = linalg.fill ins(%cst : f32)
      outs(%empty0 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %matmul0 = linalg.matmul ins(%lhs, %rhs0 : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%fill0 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %empty1 = tensor.empty() : tensor<10x40xf16>
  %trunc0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul0 : tensor<10x40xf32>) outs(%empty1 : tensor<10x40xf16>) {
    ^bb0(%b0: f32, %b1 : f16):
      %0 = arith.truncf %b0 : f32 to f16
      linalg.yield %0 : f16
  } -> tensor<10x40xf16>
  %empty2 = tensor.empty() : tensor<10x80xf32>
  %fill1 = linalg.fill ins(%cst : f32)
      outs(%empty2 : tensor<10x80xf32>) -> tensor<10x80xf32>
  %matmul1 = linalg.matmul ins(%trunc0, %rhs1 : tensor<10x40xf16>, tensor<40x80xf16>)
      outs(%fill1 : tensor<10x80xf32>) -> tensor<10x80xf32>
  %empty3 = tensor.empty() : tensor<10x80xf16>
  %trunc1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul1 : tensor<10x80xf32>) outs(%empty3 : tensor<10x80xf16>) {
    ^bb0(%b0: f32, %b1 : f16):
      %0 = arith.truncf %b0 : f32 to f16
      linalg.yield %0 : f16
  } -> tensor<10x80xf16>
  return %trunc1 : tensor<10x80xf16>
}
// CHECK-LABEL: test
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     tensor.empty
//       CHECK:     linalg.fill
//       CHECK:     linalg.matmul
//       CHECK:     linalg.generic
//       CHECK:     tensor.empty
//       CHECK:     linalg.fill
//       CHECK:     linalg.matmul
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   return %[[DISPATCH]]
