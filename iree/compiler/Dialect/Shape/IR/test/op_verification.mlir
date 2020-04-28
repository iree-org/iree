// RUN: iree-opt -split-input-file -verify-diagnostics %s

// -----
func @tie_shape_mismatch_type(%arg0 : tensor<2x?x4xf32>, %arg1 : !shapex.ranked_shape<[1]>) {
  // expected-error @+1 {{dims must match between tensor and shape}}
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<2x?x4xf32>, !shapex.ranked_shape<[1]>
  return
}

// -----
func @get_ranked_shape_same_rank(%arg0 : tensor<2x?x4xf32>) {
  // expected-error @+1 {{op operand and result must be of same rank}}
  %0 = shapex.get_ranked_shape %arg0 : tensor<2x?x4xf32> -> !shapex.ranked_shape<[2]>
  return
}

// -----
func @get_ranked_shape_not_equal_dims(%arg0 : tensor<2x?x4xf32>) {
  // expected-error @+1 {{op operand tensor and result shape must be equal}}
  %0 = shapex.get_ranked_shape %arg0 : tensor<2x?x4xf32> -> !shapex.ranked_shape<[2,2,4]>
  return
}

// -----
func @const_ranked_shape_wrong_type() {
  // expected-error @+1 {{result #0 must be Ranked shape type, but got 'i32'}}
  %0 = shapex.const_ranked_shape : i32
  return
}

// -----
func @const_ranked_shape_not_static() {
  // expected-error @+1 {{must be a fully static ranked_shape}}
  %0 = shapex.const_ranked_shape : !shapex.ranked_shape<[2,?,4]>
  return
}

// -----
func @ranked_dim_out_of_range(%arg0 : !shapex.ranked_shape<[2,4]>) {
  // expected-error @+1 {{index out of bounds of shape}}
  %0 = shapex.ranked_dim %arg0[2] : !shapex.ranked_shape<[2,4]> -> index
  return
}

// -----

func @compatible_from_extent_tensor(%arg0: tensor<1xindex>) {
  %0 = "shapex.from_extent_tensor"(%arg0) : (tensor<1xindex>) -> !shapex.ranked_shape<[3]>
  return
}
