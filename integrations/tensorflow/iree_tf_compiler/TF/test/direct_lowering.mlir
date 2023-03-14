// RUN: iree-tf-opt --iree-tf-convert-to-mhlo --split-input-file %s | FileCheck %s

// CHECK-LABEL: @expand_dims
func.func @expand_dims(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?x1xf32> {
  // CHECK: %[[R:.*]] = tensor.expand_shape %arg0 {{\[}}[0], [1], [2, 3]] : tensor<?x?x?xf32> into tensor<?x?x?x1xf32>
  // CHECK: return %[[R]]
  %axis = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> (tensor<i32>)
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<?x?x?xf32>, tensor<i32>) -> (tensor<?x?x?x1xf32>)
  return %0 : tensor<?x?x?x1xf32>
}

// -----
// CHECK-LABEL: @expand_dims_mismatch
// Verifies that the fallback lowering to reshape is used if the static
// information in the shape does not match the request expansion dim.
func.func @expand_dims_mismatch(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: mhlo.dynamic_reshape
  %axis = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> (tensor<i32>)
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<?x?x?xf32>, tensor<i32>) -> (tensor<?x?x?x?xf32>)
  return %0 : tensor<?x?x?x?xf32>
}

// -----
// CHECK-LABEL: @squeeze
func.func @squeeze(%arg0 : tensor<?x1x1x1001xf32>) -> tensor<?x1001xf32> {
  // CHECK: %[[R:.*]] = tensor.collapse_shape %arg0 {{\[}}[0], [1, 2, 3]] : tensor<?x1x1x1001xf32> into tensor<?x1001xf32>
  // CHECK: return %[[R]]
  %0 = "tf.Squeeze"(%arg0) {device = "", squeeze_dims = [1, 2]} : (tensor<?x1x1x1001xf32>) -> tensor<?x1001xf32>
  return %0 : tensor<?x1001xf32>
}
