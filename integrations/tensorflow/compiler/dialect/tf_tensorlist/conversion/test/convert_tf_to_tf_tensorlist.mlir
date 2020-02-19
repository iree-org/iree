// RUN: iree-tf-opt %s -pass-pipeline=convert-tf-to-tf_tensorlist,canonicalize | IreeFileCheck %s

// TODO(silvasean): Do conversion in a better way that doesn't require this canonicalization cleanup.
// TODO(silvasean): Handle multiple basic blocks.
// TODO(silvasean): Handle interprocedural conversion.
// TODO(silvasean): Test more cases.

// CHECK-LABEL:  func @identity_through_tensorlist(%arg0: tensor<f32>) -> tensor<f32> {
// CHECK:          %cst = constant dense<0> : tensor<i32>
// CHECK:          %cst_0 = constant dense<[]> : tensor<0xi32>
// CHECK:          %cst_1 = constant dense<1> : tensor<i32>
// CHECK:          %0 = "tf_tensorlist.Reserve"(%cst_0, %cst_1) : (tensor<0xi32>, tensor<i32>) -> !tf_tensorlist.list
// CHECK:          %1 = "tf_tensorlist.SetItem"(%0, %cst, %arg0) : (!tf_tensorlist.list, tensor<i32>, tensor<f32>) -> !tf_tensorlist.list
// CHECK:          %2 = "tf_tensorlist.GetItem"(%1, %cst, %cst_0) : (!tf_tensorlist.list, tensor<i32>, tensor<0xi32>) -> tensor<f32>
// CHECK:          return %2 : tensor<f32>

func @identity_through_tensorlist(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = constant dense<0> : tensor<i32>
  %1 = constant dense<[]> : tensor<0xi32>
  %2 = constant dense<1> : tensor<i32>
  %3 = "tf.TensorListReserve"(%1, %2) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<f32>>>
  %4 = "tf.TensorListSetItem"(%3, %0, %arg0) : (tensor<!tf.variant<tensor<f32>>>, tensor<i32>, tensor<f32>) -> tensor<!tf.variant<tensor<f32>>>
  %5 = "tf.TensorListGetItem"(%4, %0, %1) : (tensor<!tf.variant<tensor<f32>>>, tensor<i32>, tensor<0xi32>) -> tensor<f32>
  return %5 : tensor<f32>
}
