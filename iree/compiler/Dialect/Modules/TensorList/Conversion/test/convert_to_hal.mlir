
// RUN: iree-opt --iree-convert-to-hal %s --split-input-file | IreeFileCheck %s

// CHECK: @Reserve
func @Reserve(%arg0: tensor<0xi32>, %arg1: tensor<i32>) -> !tensorlist.list {
  // CHECK: %c16777248_i32 = constant 16777248 : i32
  // CHECK: %[[VIEW0:.+]] = hal.buffer_view.create %arg0, element_type = %c16777248_i32, shape = [%c0]
  // CHECK: %c16777248_i32_0 = constant 16777248 : i32
  // CHECK: %[[VIEW1:.+]] = hal.buffer_view.create %arg1, element_type = %c16777248_i32_0, shape = []
  // CHECK: %[[LIST:.+]] = "tensorlist.Reserve"(%[[VIEW0]], %[[VIEW1]]) {element_type = 50331680 : i32}
  %0 = "tensorlist.Reserve.Tensor"(%arg0, %arg1) {element_type = f32} : (tensor<0xi32>, tensor<i32>) -> !tensorlist.list

  // CHECK: return %[[LIST]]
  return %0 : !tensorlist.list
}

// -----

// CHECK: @FromTensorList
func @FromTensorList(%arg0: tensor<10xi32>) -> !tensorlist.list {
  // CHECK: [[VIEW:%.+]] = hal.buffer_view.create %arg0
  // CHECK: [[RET:%.+]] = "tensorlist.FromTensor"([[VIEW]])
  %0 = "tensorlist.FromTensor"(%arg0) : (tensor<10xi32>) -> !tensorlist.list

  // CHECK: return [[RET]]
  return %0 : !tensorlist.list
}

// -----

// CHECK: @SetItem
func @SetItem(%arg0: !tensorlist.list, %arg1: tensor<i32>, %arg2: tensor<f32>) -> !tensorlist.list {
  // CHECK: [[VIEW0:%.+]] = hal.buffer_view.create %arg1
  // CHECK: [[VIEW1:%.+]] = hal.buffer_view.create %arg2
  // CHECK: [[RET:%.+]] = "tensorlist.SetItem"(%arg0, [[VIEW0]], [[VIEW1]])

  %0 = "tensorlist.SetItem"(%arg0, %arg1, %arg2) : (!tensorlist.list, tensor<i32>, tensor<f32>) -> !tensorlist.list
  // CHECK: return [[RET]]
  return %0 : !tensorlist.list
}

// -----

// CHECK: @GetItem
func @GetItem(%arg0: !tensorlist.list, %arg1: tensor<i32>) -> tensor<f32> {
  // CHECK: [[VIEW:%.+]] = hal.buffer_view.create %arg1
  // CHECK: [[ITEM:%.+]] = "tensorlist.GetItem"(%arg0, [[VIEW]])
  // CHECK: [[BUF:%.+]] = hal.buffer_view.buffer [[ITEM]]
  %0 = "tensorlist.GetItem"(%arg0, %arg1) : (!tensorlist.list, tensor<i32>) -> tensor<f32>

  // CHECK: return [[BUF]]
  return %0 : tensor<f32>
}

// -----

// CHECK: @Stack
func @Stack(%arg0: !tensorlist.list, %arg1: tensor<i32>) -> tensor<1xf32> {
  // CHECK-DAG: [[DEV:%.+]] = hal.ex.shared_device
  // CHECK-DAG: [[ALL:%.+]] = hal.device.allocator [[DEV]]
  // CHECK-DAG: [[VIEW:%.+]] = hal.buffer_view.create %arg1
  // CHECK-DAG: [[RES:%.+]] = "tensorlist.Stack"([[ALL]], %arg0, [[VIEW]])
  // CHECK-DAG: [[BUF:%.+]] = hal.buffer_view.buffer [[RES]]
  %0 = "tensorlist.Stack.Tensor"(%arg0, %arg1) : (!tensorlist.list, tensor<i32>) -> tensor<1xf32>

  // CHECK: return [[BUF]]
  return %0 : tensor<1xf32>
}

// -----

// CHECK: @Concat
func @Concat(%arg0: !tensorlist.list) -> tensor<1xf32> {
  // CHECK: [[DEV:%.+]] = hal.ex.shared_device : !hal.device
  // CHECK: [[ALL:%.+]] = hal.device.allocator [[DEV]]
  // CHECK: [[RES:%.+]] = "tensorlist.Concat"([[ALL]], %arg0)
  // CHECK: [[BUF:%.+]] = hal.buffer_view.buffer [[RES]]
  %0 = "tensorlist.Concat.Tensor"(%arg0) : (!tensorlist.list) -> tensor<1xf32>

  // CHECK: return [[BUF]]
  return %0 : tensor<1xf32>
}

