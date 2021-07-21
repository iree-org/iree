// RUN: iree-opt -iree-llvmcpu-plan-conv-loop-order %s | IreeFileCheck %s

func @conv(%filter: memref<3x3x3x32xf32>, %input: memref<1x225x225x3xf32>, %output: memref<1x112x112x32xf32>) {
  linalg.conv(%filter, %input, %output) {dilations = [1, 1], strides = [2, 2]} : memref<3x3x3x32xf32>, memref<1x225x225x3xf32>, memref<1x112x112x32xf32>
  return
}

// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
// CHECK:  #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d3, d2 * 2 + d4, d5)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[FILTER_MAP]], #[[INPUT_MAP]], #[[OUTPUT_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "window", "window", "reduction", "parallel"]


