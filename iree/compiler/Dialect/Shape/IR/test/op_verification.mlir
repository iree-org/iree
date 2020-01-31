// RUN: iree-opt -split-input-file -verify-diagnostics %s

// -----
func @tie_shape_mismatch_type(%arg0 : tensor<2x?x4xf32>, %arg1 : !shape.ranked_shape<[1],i32>) {
  // expected-error @+1 {{dims must match between tensor and shape}}
  %0 = shape.tie_shape %arg0, %arg1 : tensor<2x?x4xf32>, !shape.ranked_shape<[1],i32>
  return
}

// -----
func @get_ranked_shape_same_rank(%arg0 : tensor<2x?x4xf32>) {
  // expected-error @+1 {{op operand and result must be of same rank}}
  %0 = shape.get_ranked_shape %arg0 : tensor<2x?x4xf32> -> !shape.ranked_shape<[2],i32>
  return
}

// -----
func @get_ranked_shape_not_equal_dims(%arg0 : tensor<2x?x4xf32>) {
  // expected-error @+1 {{op operand tensor and result shape must be equal}}
  %0 = shape.get_ranked_shape %arg0 : tensor<2x?x4xf32> -> !shape.ranked_shape<[2,2,4],i32>
  return
}

// -----
func @const_ranked_shape_wrong_type() {
  // expected-error @+1 {{result #0 must be Ranked shape type, but got 'i32'}}
  %0 = shape.const_ranked_shape : i32
  return
}

// -----
func @const_ranked_shape_not_static() {
  // expected-error @+1 {{must be a fully static ranked_shape}}
  %0 = shape.const_ranked_shape : !shape.ranked_shape<[2,?,4],i32>
  return
}

// -----
func @ranked_dim_out_of_range(%arg0 : !shape.ranked_shape<[2,4],i32>) {
  // expected-error @+1 {{index out of bounds of shape}}
  %0 = shape.ranked_dim %arg0[2] : !shape.ranked_shape<[2,4],i32>
  return
}

// -----
// expected-note @+1 {{prior use here}}
func @make_ranked_shape_dim_type_mismatch(%arg0 : index, %arg1 : index) {
  // expected-error @+1 {{use of value '%arg0' expects different type than prior uses: 'i32' vs 'index'}}
  %0 = shape.make_ranked_shape %arg0, %arg1 -> !shape.ranked_shape<[1,?,?,16],i32>
  return
}
