// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// -----
// CHECK-LABEL: @parse_print_get_ranked_shape
func @parse_print_get_ranked_shape(%arg0 : tensor<2x?x4xi32>) {
  // CHECK: shape.get_ranked_shape %arg0 : tensor<2x?x4xi32> -> !shape.ranked_shape<2x?x4xi32>
  %0 = shape.get_ranked_shape %arg0 : tensor<2x?x4xi32> -> !shape.ranked_shape<2x?x4xi32>
  return
}
