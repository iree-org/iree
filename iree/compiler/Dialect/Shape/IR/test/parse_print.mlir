// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// -----
// CHECK-LABEL: @const_ranked_shape
func @const_ranked_shape() -> !shapex.ranked_shape<[2,4]> {
  // CHECK: %rs2_4 = shapex.const_ranked_shape : !shapex.ranked_shape<[2,4]>
  %0 = shapex.const_ranked_shape : !shapex.ranked_shape<[2,4]>
  // CHECK: %rs = shapex.const_ranked_shape : !shapex.ranked_shape<[]>
  %1 = shapex.const_ranked_shape : !shapex.ranked_shape<[]>
  // CHECK: %rs5_6 = shapex.const_ranked_shape : !shapex.ranked_shape<[5,6]>
  %2 = shapex.const_ranked_shape : !shapex.ranked_shape<[5,6]>
  return %0 : !shapex.ranked_shape<[2,4]>
}

// -----
// CHECK-LABEL: @ranked_dim
func @ranked_dim(%arg0 : !shapex.ranked_shape<[2,4]>)  {
  // CHECK: shapex.ranked_dim %arg0[1] : !shapex.ranked_shape<[2,4]> -> index
  %0 = shapex.ranked_dim %arg0[1] : !shapex.ranked_shape<[2,4]> -> index
  return
}

// -----
// CHECK-LABEL: @make_ranked_shape
func @make_ranked_shape(%arg0 : index, %arg1 : index) -> (!shapex.ranked_shape<[1,?,?,16]>) {
  // CHECK: shapex.make_ranked_shape %arg0, %arg1 : (index, index) -> !shapex.ranked_shape<[1,?,?,16]>
  %0 = shapex.make_ranked_shape %arg0, %arg1 : (index, index) -> !shapex.ranked_shape<[1,?,?,16]>
  return %0 : !shapex.ranked_shape<[1,?,?,16]>
}
