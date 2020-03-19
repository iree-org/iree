// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-hoist-shape-calculations %s | IreeFileCheck %s

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: index) {
  // CHECK: shapex.make_ranked_shape
  // CHECK: addf
  %t = addf %arg0, %arg0 : tensor<?xf32>
  %shape = shapex.make_ranked_shape %arg1 -> !shapex.ranked_shape<[?]>
  shapex.tie_shape %t, %shape : tensor<?xf32>, !shapex.ranked_shape<[?]>
  return
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>) {
  // CHECK: addf
  // CHECK: dim
  // CHECK: shapex.make_ranked_shape
  %t = addf %arg0, %arg0 : tensor<?xf32>
  %dim = dim %t, 0 : tensor<?xf32>
  %shape = shapex.make_ranked_shape %dim -> !shapex.ranked_shape<[?]>
  shapex.tie_shape %t, %shape : tensor<?xf32>, !shapex.ranked_shape<[?]>
  return
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: i32) {
  // CHECK: addi
  // CHECK: muli
  // CHECK: shapex.make_ranked_shape
  // CHECK: some_dialect.some_op
  "some_dialect.some_op"() : () -> ()
  %addi = addi %arg1, %arg1 : i32
  %dim = muli %addi, %addi : i32
  %shape = shapex.make_ranked_shape %dim -> !shapex.ranked_shape<[?], i32>
  shapex.tie_shape %arg0, %shape : tensor<?xf32>, !shapex.ranked_shape<[?], i32>
  return
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: index) {
  // CHECK: some_dialect.some_op
  // CHECK: some_dialect.side_effecting_muli
  // CHECK: shapex.make_ranked_shape
  "some_dialect.some_op"() : () -> ()
  %dim = "some_dialect.side_effecting_muli"(%arg1, %arg1) : (index, index) -> index
  %shape = shapex.make_ranked_shape %dim -> !shapex.ranked_shape<[?]>
  shapex.tie_shape %arg0, %shape : tensor<?xf32>, !shapex.ranked_shape<[?]>
  return
}
