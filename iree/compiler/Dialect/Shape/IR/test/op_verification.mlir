// RUN: iree-opt -split-input-file -verify-diagnostics %s

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
