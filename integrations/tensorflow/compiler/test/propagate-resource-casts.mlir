// RUN: iree-tf-opt -split-input-file -verify-diagnostics -iree-propagate-resource-casts <%s | IreeFileCheck %s

// CHECK-LABEL: @noop_cast
func @noop_cast(%arg0: tensor<!tf.resource>) -> tensor<*xi16> {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource>) -> (tensor<!tf.resource>)
  // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource>)
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource>) -> tensor<*xi16>
  return %1 : tensor<*xi16>
}

// -----

// CHECK-LABEL: @simple_bypass
func @simple_bypass(%arg0: tensor<!tf.resource<tensor<*xi16>>>) -> tensor<*xi16> {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource<tensor<*xi16>>>) -> (tensor<!tf.resource>)
  // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<*xi16>>>)
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource>) -> tensor<*xi16>
  return %1 : tensor<*xi16>
}

// -----

// CHECK-LABEL: @simple_no_bypass
func @simple_no_bypass(%arg0: tensor<!tf.resource>) -> tensor<*xi16> {
  // CHECK: [[V:%.+]] = "tf.Cast"(%arg0)
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource>) -> (tensor<!tf.resource<tensor<*xi16>>>)
  // CHECK: "tf.ReadVariableOp"([[V]]) : (tensor<!tf.resource<tensor<*xi16>>>)
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource<tensor<*xi16>>>) -> tensor<*xi16>
  return %1 : tensor<*xi16>
}

// -----

// CHECK-LABEL: @dynamic_bypass
func @dynamic_bypass(%arg0: tensor<!tf.resource<tensor<?xi16>>>) -> tensor<?xi16> {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource<tensor<?xi16>>>) -> (tensor<!tf.resource>)
  // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<?xi16>>>)
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource>) -> tensor<?xi16>
  return %1 : tensor<?xi16>
}

// -----

// CHECK-LABEL: @dynamic_no_bypass
func @dynamic_no_bypass(%arg0: tensor<!tf.resource>) -> tensor<?xi16> {
  // CHECK: [[V:%.+]] = "tf.Cast"(%arg0)
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource>) -> (tensor<!tf.resource<tensor<?xi16>>>)
  // CHECK: "tf.ReadVariableOp"([[V]]) : (tensor<!tf.resource<tensor<?xi16>>>)
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource<tensor<?xi16>>>) -> tensor<?xi16>
  return %1 : tensor<?xi16>
}

// -----

// CHECK-LABEL: @static_bypass
func @static_bypass(%arg0: tensor<!tf.resource<tensor<5xi16>>>) -> tensor<5xi16> {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource<tensor<5xi16>>>) -> (tensor<!tf.resource<tensor<?xi16>>>)
  // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<5xi16>>>)
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource<tensor<?xi16>>>) -> tensor<5xi16>
  return %1 : tensor<5xi16>
}

// -----

// CHECK-LABEL: @static_bypass_to_unranked
func @static_bypass_to_unranked(%arg0: tensor<!tf.resource<tensor<5xi16>>>) -> tensor<*xi16> {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource<tensor<5xi16>>>) -> (tensor<!tf.resource<tensor<*xi16>>>)
  // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<5xi16>>>)
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource<tensor<*xi16>>>) -> tensor<*xi16>
  return %1 : tensor<*xi16>
}

// -----

// CHECK-LABEL: @static_no_bypass
func @static_no_bypass(%arg0: tensor<!tf.resource<tensor<?xi16>>>) -> tensor<5xi16> {
  // CHECK: [[V:%.+]] = "tf.Cast"(%arg0)
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource<tensor<?xi16>>>) -> (tensor<!tf.resource<tensor<5xi16>>>)
  // CHECK: "tf.ReadVariableOp"([[V]]) : (tensor<!tf.resource<tensor<5xi16>>>)
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf.resource<tensor<5xi16>>>) -> tensor<5xi16>
  return %1 : tensor<5xi16>
}
