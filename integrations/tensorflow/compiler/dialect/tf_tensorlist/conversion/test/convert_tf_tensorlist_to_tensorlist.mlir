// RUN: iree-tf-opt <%s -iree-tf-tensorlist-convert-to-tensorlist -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @Reserve
func @Reserve(%arg0: tensor<0xi32>, %arg1: tensor<i32>) -> !tf_tensorlist.list{
// CHECK:         "tensorlist.Reserve.Tensor"(%arg0, %arg1)
  %0 = "tf_tensorlist.Reserve"(%arg0, %arg1) {element_type = f32} : (tensor<0xi32>, tensor<i32>) -> !tf_tensorlist.list
  return %0 : !tf_tensorlist.list
}

// -----

// CHECK-LABEL: func @FromTensorList
func @FromTensorList(%arg0: tensor<10xi32>) -> !tf_tensorlist.list{
// CHECK:         "tensorlist.FromTensor"(%arg0)
  %0 = "tf_tensorlist.FromTensor"(%arg0) : (tensor<10xi32>) -> !tf_tensorlist.list
  return %0 : !tf_tensorlist.list
}

// -----

// CHECK-LABEL: func @SetItem
func @SetItem(%arg0: !tf_tensorlist.list, %arg1: tensor<i32>, %arg2: tensor<f32>) -> !tf_tensorlist.list{
// CHECK:         "tensorlist.SetItem"(%arg0, %arg1, %arg2)
  %0 = "tf_tensorlist.SetItem"(%arg0, %arg1, %arg2) : (!tf_tensorlist.list, tensor<i32>, tensor<f32>) -> !tf_tensorlist.list
  return %0 : !tf_tensorlist.list
}

// -----

// CHECK-LABEL: func @GetItem
func @GetItem(%arg0: !tf_tensorlist.list, %arg1: tensor<i32>) -> tensor<f32> {
// CHECK:         "tensorlist.GetItem"(%arg0, %arg1)
  %0 = "tf_tensorlist.GetItem"(%arg0, %arg1) : (!tf_tensorlist.list, tensor<i32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @Stack
func @Stack(%arg0: !tf_tensorlist.list, %arg1: tensor<i32>) -> tensor<1xf32> {
// CHECK:         "tensorlist.Stack.Tensor"(%arg0, %arg1)
  %0 = "tf_tensorlist.Stack"(%arg0, %arg1) : (!tf_tensorlist.list, tensor<i32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
 
// -----

// CHECK-LABEL: func @Concat
func @Concat(%arg0: !tf_tensorlist.list) -> (tensor<1xf32>) {
// CHECK:         "tensorlist.Concat.Tensor"(%arg0)
  %0 = "tf_tensorlist.Concat"(%arg0) : (!tf_tensorlist.list) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
