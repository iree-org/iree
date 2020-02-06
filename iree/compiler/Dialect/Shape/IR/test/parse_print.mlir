// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// -----
// CHECK-LABEL: @parse_print_tie_shape
func @parse_print_tie_shape(%arg0 : tensor<2x?x4xf32>, %arg1 : !shapex.ranked_shape<[2,?,4],i32>) {
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<2x?x4xf32>, !shapex.ranked_shape<[2,?,4],i32>
  return
}


// -----
// CHECK-LABEL: @parse_print_get_ranked_shape
func @parse_print_get_ranked_shape(%arg0 : tensor<2x?x4xi32>) {
  // CHECK: shapex.get_ranked_shape %arg0 : tensor<2x?x4xi32> -> !shapex.ranked_shape<[2,?,4],i32>
  %0 = shapex.get_ranked_shape %arg0 : tensor<2x?x4xi32> -> !shapex.ranked_shape<[2,?,4],i32>
  return
}

// -----
// CHECK-LABEL: @const_ranked_shape
func @const_ranked_shape() -> !shapex.ranked_shape<[2,4],i32> {
  // CHECK: shapex.const_ranked_shape : !shapex.ranked_shape<[2,4],i32>
  %0 = shapex.const_ranked_shape : !shapex.ranked_shape<[2,4],i32>
  return %0 : !shapex.ranked_shape<[2,4],i32>
}

// -----
// CHECK-LABEL: @ranked_dim
func @ranked_dim(%arg0 : !shapex.ranked_shape<[2,4],i32>)  {
  // CHECK: shapex.ranked_dim %arg0[1] : !shapex.ranked_shape<[2,4],i32>
  %0 = shapex.ranked_dim %arg0[1] : !shapex.ranked_shape<[2,4],i32>
  return
}

// -----
// CHECK-LABEL: @ranked_dims
func @ranked_dims(%arg0 : !shapex.ranked_shape<[2,4],i32>)  {
  // CHECK: shapex.ranked_dims %arg0 : !shapex.ranked_shape<[2,4],i32>
  %0:2 = shapex.ranked_dims %arg0 : !shapex.ranked_shape<[2,4],i32>
  return
}

// -----
// CHECK-LABEL: @cast_compatible_shape
func @cast_compatible_shape(%arg0 : !shapex.ranked_shape<[2,4],i32>, %arg1 : !shapex.ranked_shape<[2,4],i32>)  {
  // CHECK: shapex.cast_compatible_shape %arg0, %arg1 :
  // CHECK-SAME: !shapex.ranked_shape<[2,4],i32>, !shapex.ranked_shape<[2,4],i32> ->
  // CHECK-SAME: !shapex.ranked_shape<[2,4],i32>
  %0 = shapex.cast_compatible_shape %arg0, %arg1 : !shapex.ranked_shape<[2,4],i32>, !shapex.ranked_shape<[2,4],i32> ->
      !shapex.ranked_shape<[2,4],i32>
  return
}

// -----
// CHECK-LABEL: @make_ranked_shape
func @make_ranked_shape(%arg0 : index, %arg1 : index) -> (!shapex.ranked_shape<[1,?,?,16]>) {
  // CHECK: shapex.make_ranked_shape %arg0, %arg1 -> !shapex.ranked_shape<[1,?,?,16]>
  %0 = shapex.make_ranked_shape %arg0, %arg1 -> !shapex.ranked_shape<[1,?,?,16]>
  return %0 : !shapex.ranked_shape<[1,?,?,16]>
}
