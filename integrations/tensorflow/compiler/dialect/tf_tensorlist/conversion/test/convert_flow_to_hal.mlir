// RUN: iree-tf-opt <%s -iree-convert-to-hal -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @Reserve(%arg0: !hal.buffer, %arg1: !hal.buffer) -> !tensorlist.list {
func @Reserve(%arg0: tensor<0xi32>, %arg1: tensor<i32>) -> !tf_tensorlist.list{
// CHECK:         [[VIEW0:%.+]] = hal.buffer_view.create %arg0{{.*}}
// CHECK:         [[VIEW1:%.+]] = hal.buffer_view.create %arg1{{.*}}
// CHECK:         "tensorlist.Reserve"([[VIEW0]], [[VIEW1]])
  %0 = "tf_tensorlist.Reserve"(%arg0, %arg1) : (tensor<0xi32>, tensor<i32>) -> !tf_tensorlist.list
  return %0 : !tf_tensorlist.list
}

// CHECK-LABEL: func @SetItem(%arg0: !tensorlist.list, %arg1: !hal.buffer, %arg2: !hal.buffer) -> !tensorlist.list {
func @SetItem(%arg0: !tf_tensorlist.list, %arg1: tensor<i32>, %arg2: tensor<f32>) -> !tf_tensorlist.list{
// CHECK:         [[VIEW1:%.+]]  = hal.buffer_view.create %arg1{{.*}}
// CHECK:         [[VIEW2:%.+]]  = hal.buffer_view.create %arg2{{.*}}
// CHECK:         "tensorlist.SetItem"(%arg0, [[VIEW1]], [[VIEW2]])
  %0 = "tf_tensorlist.SetItem"(%arg0, %arg1, %arg2) : (!tf_tensorlist.list, tensor<i32>, tensor<f32>) -> !tf_tensorlist.list
  return %0 : !tf_tensorlist.list
}

// CHECK-LABEL: func @GetItem(%arg0: !tensorlist.list, %arg1: !hal.buffer, %arg2: !hal.buffer) -> !hal.buffer {
func @GetItem(%arg0: !tf_tensorlist.list, %arg1: tensor<i32>, %arg2: tensor<0xi32>) -> tensor<f32> {
// CHECK:         [[VIEW1:%.+]] = hal.buffer_view.create %arg1{{.*}}
// CHECK:         [[VIEW2:%.+]] = hal.buffer_view.create %arg2{{.*}}
// CHECK:         "tensorlist.GetItem"(%arg0, [[VIEW1]], [[VIEW2]])
  %0 = "tf_tensorlist.GetItem"(%arg0, %arg1, %arg2) : (!tf_tensorlist.list, tensor<i32>, tensor<0xi32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @Stack(%arg0: !tensorlist.list, %arg1: !hal.buffer, %arg2: !hal.buffer) -> !hal.buffer {
func @Stack(%arg0: !tf_tensorlist.list, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> tensor<1xf32> {
// CHECK:         [[VIEW1:%.+]] = hal.buffer_view.create %arg1{{.*}}
// CHECK:         [[VIEW2:%.+]] = hal.buffer_view.create %arg2{{.*}}
// CHECK:         "tensorlist.Stack"(%arg0, [[VIEW1]], [[VIEW2]])
  %0 = "tf_tensorlist.Stack"(%arg0, %arg1, %arg2) : (!tf_tensorlist.list, tensor<1xi32>, tensor<i32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
