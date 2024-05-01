// RUN: iree-opt --split-input-file --iree-flow-elementwise-op-fusion %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
util.func @output_transpose_map(%arg0: tensor<2x128x128x320xf32>) -> tensor<2x320x128x128xf16> {
  %0 = tensor.empty() : tensor<2x320x128x128xf16>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x128x128x320xf32>) outs(%0 : tensor<2x320x128x128xf16>) {
  ^bb0(%in: f32, %out: f16):
    %2 = arith.truncf %in : f32 to f16
    linalg.yield %2 : f16
  } -> tensor<2x320x128x128xf16>
  util.return %1 : tensor<2x320x128x128xf16>
}

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @output_transpose_map
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
