// RUN: iree-opt -convert-shape-to-shapex -split-input-file -verify-diagnostics -allow-unregistered-dialect <%s | IreeFileCheck %s

// -----
// shape.const_shape
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>) {
  // CHECK: shapex.const_ranked_shape : !shapex.ranked_shape<[1,2,3]>
  %0 = shape.const_shape [1, 2, 3] : tensor<?xindex>
  "foo.use"(%0) : (tensor<?xindex>) -> ()
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
  %index = constant 0 : index
  // CHECK: %[[RS:.+]] = shapex.get_ranked_shape %arg0
  // CHECK: %[[HEAD:.+]] = "shapex.gather_extents"(%[[RS]]) {indices = dense<> : tensor<0xi64>} : (!shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[]>
  // CHECK: %[[TAIL:.+]] = "shapex.gather_extents"(%[[RS]]) {indices = dense<0> : tensor<1xi64>} : (!shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[?]>
  // CHECK: "foo.use"(%[[HEAD]], %[[TAIL]])
  %head, %tail = "shape.split_at"(%0, %index) : (!shape.shape, index) -> (!shape.shape, !shape.shape)
  "foo.use"(%head, %tail) : (!shape.shape, !shape.shape) -> ()
  return
}

// -----
// No conversion -- index is dynamic.
func @f(%arg0: tensor<?xf32>, %index: index) {
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape
  // expected-error @+1 {{failed to legalize operation}}
  %head, %tail = "shape.split_at"(%0, %index) : (!shape.shape, index) -> (!shape.shape, !shape.shape)
  "foo.use"(%head, %tail) : (!shape.shape, !shape.shape) -> ()
  return
}

// -----
// shape.get_extent
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>) {
  %c0 = constant 0 : index
  // CHECK: %[[SHAPE:.+]] = shapex.get_ranked_shape %arg0 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape

  // CHECK: [[DIM:%.+]] = shapex.ranked_dim %[[SHAPE]][0] : !shapex.ranked_shape<[?]> -> index
  %result = shape.get_extent %0, %c0 : !shape.shape, index -> !shape.size
  return
}

// -----
// shape.from_extents
// CHECK-LABEL: func @f
func @f(%arg0: index) {
  // CHECK: shapex.make_ranked_shape %arg0, %arg0
  %result = "shape.from_extents"(%arg0, %arg0) : (index, index) -> !shape.shape
  return
}

// -----
// shape.from_extent_tensor
// CHECK-LABEL: func @f
func @f(%arg0: tensor<3xindex>) {
  // CHECK: "shapex.from_extent_tensor"(%arg0) : (tensor<3xindex>) -> !shapex.ranked_shape<[?,?,?]>
  %result = "shape.from_extent_tensor"(%arg0) : (tensor<3xindex>)
      -> !shape.shape
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
// shape.broadcast
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) {
  // CHECK: %[[LHSRS:.+]] = shapex.get_ranked_shape %arg0
  // CHECK: %[[RHSRS:.+]] = shapex.get_ranked_shape %arg1
  // CHECK: %[[BROADCASTED:.+]] = "shapex.ranked_broadcast_shape"(%[[LHSRS]], %[[RHSRS]])
  // CHECK-SAME: : (!shapex.ranked_shape<[?,?]>, !shapex.ranked_shape<[?,?]>) -> !shapex.ranked_shape<[?,?]>
  // CHECK: "use"(%[[BROADCASTED]])
  %0 = "shape.shape_of"(%arg0) : (tensor<?x?xf32>) -> tensor<2xindex>
  %1 = "shape.shape_of"(%arg1) : (tensor<?x?xf32>) -> tensor<2xindex>
  %2 = "shape.broadcast"(%0, %1) : (tensor<2xindex>, tensor<2xindex>) -> tensor<2xindex>
  "use"(%2) : (tensor<2xindex>) -> ()
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

// -----
// tensor.cast
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: %[[LHSRS:.+]] = shapex.get_ranked_shape %arg0 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  // CHECK: %[[RHSRS:.+]] = shapex.get_ranked_shape %arg1 : tensor<?xf32> -> !shapex.ranked_shape<[?]>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<?xindex>
  // CHECK: %[[BROADCASTED:.+]] = "shapex.ranked_broadcast_shape"(%[[LHSRS]], %[[RHSRS]]) {
  // CHECK-SAME: lhs_broadcast_dimensions = dense<0> : tensor<1xi64>,
  // CHECK-SAME: rhs_broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK-SAME: : (!shapex.ranked_shape<[?]>, !shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[?]>
  %2 = shape.broadcast %0, %1 : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  // CHECK: %[[EXTENTS:.+]] = "shapex.to_extent_tensor"(%[[BROADCASTED]]) : (!shapex.ranked_shape<[?]>) -> tensor<1xindex>
  %3 = tensor.cast %2 : tensor<?xindex> to tensor<1xindex>
  // CHECK: "foo.use"(%[[EXTENTS]])
  "foo.use"(%3) : (tensor<1xindex>) -> ()
  return
}

// -----
// Handling of existing shapex.from_extent_tensor ops.
// CHECK-LABEL:   func @f
func @f(%arg0: tensor<?x3x3x3x3xf32>, %arg1: tensor<?x3x3x3x3xf32>) -> (tensor<?x3x3x3x3xf32>) {
  %0 = shape.shape_of %arg0 : tensor<?x3x3x3x3xf32> -> tensor<5xindex>
  %1 = shape.shape_of %arg1 : tensor<?x3x3x3x3xf32> -> tensor<5xindex>
  %2 = shape.broadcast %0, %1 : tensor<5xindex>, tensor<5xindex> -> tensor<5xindex>
  // CHECK-NOT: from_extent_tensor
  %3 = "shapex.from_extent_tensor"(%2) : (tensor<5xindex>) -> !shapex.ranked_shape<[?,?,?,?,?]>
  %4 = "shapex.ranked_broadcast_in_dim"(%arg0, %3) {broadcast_dimensions = dense<[0, 1, 2, 3, 4]> : tensor<5xi64>} : (tensor<?x3x3x3x3xf32>, !shapex.ranked_shape<[?,?,?,?,?]>) -> tensor<?x3x3x3x3xf32>
  %5 = "shapex.from_extent_tensor"(%2) : (tensor<5xindex>) -> !shapex.ranked_shape<[?,?,?,?,?]>
  %6 = "shapex.ranked_broadcast_in_dim"(%arg1, %5) {broadcast_dimensions = dense<[0, 1, 2, 3, 4]> : tensor<5xi64>} : (tensor<?x3x3x3x3xf32>, !shapex.ranked_shape<[?,?,?,?,?]>) -> tensor<?x3x3x3x3xf32>
  %7 = mhlo.add %4, %6 : tensor<?x3x3x3x3xf32>
  return %7 : tensor<?x3x3x3x3xf32>
}
