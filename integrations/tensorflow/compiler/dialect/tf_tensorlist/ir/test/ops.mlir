// RUN: iree-tf-opt %s | iree-tf-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @f
func @f(
  %num_elements: tensor<i32>,
  %element_shape: tensor<1xi32>,
  %list: !tf_tensorlist.list,
  %index: tensor<i32>,
  %item: tensor<?xf32>
) {
  // CHECK: tf_tensorlist.Reserve
  %2 = "tf_tensorlist.Reserve"(%element_shape, %num_elements) { element_type = f32 } : (tensor<1xi32>, tensor<i32>) -> !tf_tensorlist.list
  // CHECK: tf_tensorlist.GetItem
  %3 = "tf_tensorlist.GetItem"(%list, %index) : (!tf_tensorlist.list, tensor<i32>) -> tensor<?xf32>
  // CHECK: tf_tensorlist.SetItem
  %4 = "tf_tensorlist.SetItem"(%list, %index, %item) : (!tf_tensorlist.list, tensor<i32>, tensor<?xf32>) -> !tf_tensorlist.list
  // CHECK: tf_tensorlist.Stack
  %5 = "tf_tensorlist.Stack"(%list, %index) : (!tf_tensorlist.list, tensor<i32>) -> tensor<1x2xf32>
  // CHECK: tf_tensorlist.Concat
  %6 = "tf_tensorlist.Concat"(%list) : (!tf_tensorlist.list) -> tensor<1x2xf32>
  // CHECK: tf_tensorlist.GetDim0
  %7 = "tf_tensorlist.GetDim0"(%list) : (!tf_tensorlist.list) -> tensor<1x2xf32>
  return
}
