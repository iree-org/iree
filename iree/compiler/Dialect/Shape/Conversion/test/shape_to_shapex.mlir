// RUN: iree-opt -convert-shape-to-shapex -split-input-file -verify-diagnostics -allow-unregistered-dialect <%s | IreeFileCheck %s

// -----
// shape.const_shape
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>) {
  // CHECK: shapex.const_ranked_shape : !shapex.ranked_shape<[1,2,3]>
  %0 = shape.const_shape [1, 2, 3]
  "foo.use"(%0) : (!shape.shape) -> ()
  return
}

// -----
// shape.shape_of
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>) {
  // CHECK: shapex.get_ranked_shape %arg0 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape
  "foo.use"(%0) : (!shape.shape) -> ()
  return
}

// -----
// shape.split_at
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>) {
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape
  %index = constant 0 : i32
  // CHECK: %[[RS:.+]] = shapex.get_ranked_shape %arg0
  // CHECK: %[[HEAD:.+]] = "shapex.gather_extents"(%[[RS]]) {indices = dense<> : tensor<0xi64>} : (!shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[]>
  // CHECK: %[[TAIL:.+]] = "shapex.gather_extents"(%[[RS]]) {indices = dense<0> : tensor<1xi64>} : (!shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[?]>
  // CHECK: "foo.use"(%[[HEAD]], %[[TAIL]])
  %head, %tail = "shape.split_at"(%0, %index) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  "foo.use"(%head, %tail) : (!shape.shape, !shape.shape) -> ()
  return
}

// -----
// No conversion -- index is dynamic.
func @f(%arg0: tensor<?xf32>, %index: i32) {
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape
  // expected-error @+1 {{failed to legalize operation}}
  %head, %tail = "shape.split_at"(%0, %index) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  "foo.use"(%head, %tail) : (!shape.shape, !shape.shape) -> ()
  return
}

// -----
// shape.broadcast
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: %[[LHSRS:.+]] = shapex.get_ranked_shape %arg0 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  // CHECK: %[[RHSRS:.+]] = shapex.get_ranked_shape %arg1 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape
  %1 = "shape.shape_of"(%arg1) : (tensor<?xf32>) -> !shape.shape
  // CHECK: %[[BROADCASTED:.+]] = "shapex.ranked_broadcast_shape"(%[[LHSRS]], %[[RHSRS]]) {
  // CHECK-SAME: lhs_broadcast_dimensions = dense<0> : tensor<1xi64>,
  // CHECK-SAME: rhs_broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK-SAME: : (!shapex.ranked_shape<[?]>, !shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[?]>
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  // CHECK: "foo.use"(%[[BROADCASTED]])
  "foo.use"(%2) : (!shape.shape) -> ()
  return
}

// -----
// shape.concat
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: %[[LHSRS:.+]] = shapex.get_ranked_shape %arg0 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  // CHECK: %[[RHSRS:.+]] = shapex.get_ranked_shape %arg1 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape
  %1 = "shape.shape_of"(%arg1) : (tensor<?xf32>) -> !shape.shape
  // CHECK: %[[CONCATTED:.+]] = "shapex.gather_extents"(%[[LHSRS]], %[[RHSRS]]) {indices = dense<[0, 1]> : tensor<2xi64>}
  %2 = "shape.concat"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  // CHECK: "foo.use"(%[[CONCATTED]])
  "foo.use"(%2) : (!shape.shape) -> ()
  return
}

// -----
// shape.to_extent_tensor
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: %[[RS:.+]] = shapex.get_ranked_shape %arg0 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape
  // CHECK: %[[EXTENTS:.+]] = "shapex.to_extent_tensor"(%[[RS]])
  %1 = "shape.to_extent_tensor"(%0) : (!shape.shape) -> tensor<1xindex>
  // CHECK: "foo.use"(%[[EXTENTS]])
  "foo.use"(%1) : (tensor<1xindex>) -> ()
  return
}

