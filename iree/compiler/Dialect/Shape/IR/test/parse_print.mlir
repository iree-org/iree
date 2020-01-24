// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// -----
// CHECK-LABEL: @parse_print_tie_shape
func @parse_print_tie_shape(%arg0 : tensor<2x?x4xf32>, %arg1 : !shape.ranked_shape<2x?x4xi32>) {
  %0 = shape.tie_shape %arg0, %arg1 : tensor<2x?x4xf32>, !shape.ranked_shape<2x?x4xi32>
  return
}


// -----
// CHECK-LABEL: @parse_print_get_ranked_shape
func @parse_print_get_ranked_shape(%arg0 : tensor<2x?x4xi32>) {
  // CHECK: shape.get_ranked_shape %arg0 : tensor<2x?x4xi32> -> !shape.ranked_shape<2x?x4xi32>
  %0 = shape.get_ranked_shape %arg0 : tensor<2x?x4xi32> -> !shape.ranked_shape<2x?x4xi32>
  return
}
