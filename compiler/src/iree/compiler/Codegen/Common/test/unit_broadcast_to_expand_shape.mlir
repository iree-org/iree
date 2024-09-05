// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-unit-broadcast-to-expand-shape, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck %s

func.func @unit_broadcast(%arg0 : tensor<8640x3200xf16>) -> tensor<1x8640x3200xf16> {
  %0 = tensor.empty() : tensor<1x8640x3200xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%arg0 : tensor<8640x3200xf16>) outs(%0 : tensor<1x8640x3200xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<1x8640x3200xf16>
  return %1 : tensor<1x8640x3200xf16>
}

// CHECK-LABEL: func.func @unit_broadcast(
//   CHECK-NOT: linalg.generic
//       CHECK:   %[[RESULT:.+]] = tensor.expand_shape
//       CHECK:   return %[[RESULT]]

// -----

func.func @unit_multiple_rank_broadcast(%arg0 : tensor<8640x3200xf16>) -> tensor<1x1x8640x3200xf16> {
  %0 = tensor.empty() : tensor<1x1x8640x3200xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  ins(%arg0 : tensor<8640x3200xf16>) outs(%0 : tensor<1x1x8640x3200xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<1x1x8640x3200xf16>
  return %1 : tensor<1x1x8640x3200xf16>
}

// CHECK-LABEL: func.func @unit_multiple_rank_broadcast(
//   CHECK-NOT: linalg.generic
//       CHECK:   %[[RESULT:.+]] = tensor.expand_shape
//       CHECK:   return %[[RESULT]]

// -----

func.func @non_unit_broadcast(%arg0 : tensor<8640x3200xf16>) -> tensor<2x8640x3200xf16> {
  %0 = tensor.empty() : tensor<2x8640x3200xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%arg0 : tensor<8640x3200xf16>) outs(%0 : tensor<2x8640x3200xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<2x8640x3200xf16>
  return %1 : tensor<2x8640x3200xf16>
}
// CHECK-LABEL: func.func @non_unit_broadcast(
//     CHECK-NOT:   %[[RESULT:.+]] = tensor.expand_shape
