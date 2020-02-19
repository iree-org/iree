// RUN: iree-tf-opt %s | iree-tf-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @f
func @f(
  %variant: tensor<!tf.variant>,
  %num_elements: tensor<i32>,
  %element_shape: tensor<1xi32>,
  %list: !tf_tensorlist.list,
  %index: tensor<i32>,
  %item: tensor<?xf32>
) {
  // CHECK: tf_tensorlist.ToVariant
  %0 = "tf_tensorlist.ToVariant"(%list) : (!tf_tensorlist.list) -> tensor<!tf.variant>
  // CHECK: tf_tensorlist.FromVariant
  %1 = "tf_tensorlist.FromVariant"(%variant) : (tensor<!tf.variant>) -> !tf_tensorlist.list
  // CHECK: tf_tensorlist.Reserve
  %2 = "tf_tensorlist.Reserve"(%element_shape, %num_elements) : (tensor<1xi32>, tensor<i32>) -> !tf_tensorlist.list
  // CHECK: tf_tensorlist.GetItem
  %3 = "tf_tensorlist.GetItem"(%list, %index, %element_shape) : (!tf_tensorlist.list, tensor<i32>, tensor<1xi32>) -> tensor<?xf32>
  // CHECK: tf_tensorlist.SetItem
  %4 = "tf_tensorlist.SetItem"(%list, %index, %item) : (!tf_tensorlist.list, tensor<i32>, tensor<?xf32>) -> !tf_tensorlist.list
  return
}
