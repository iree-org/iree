// RUN: iree-opt --split-input-file --iree-wgsl-replace-push-constants %s | FileCheck %s

// CHECK-LABEL: @emptyFunctionNoOp
func.func @emptyFunctionNoOp() {
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @constantLoadIndex
func.func @constantLoadIndex() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:index>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:index> -> tensor<index>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][] : tensor<index>
  %0 = hal.interface.constant.load[0] : index
  // CHECK: = arith.index_cast %[[EXTRACT]] : index to i32
  %1 = arith.index_cast %0 : index to i32
  return
}

// -----

// CHECK-LABEL: @constantLoadI32
func.func @constantLoadI32() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:i32>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:i32> -> tensor<i32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][] : tensor<i32>
  %0 = hal.interface.constant.load[0] : i32
  // CHECK: = arith.index_cast %[[EXTRACT]] : i32 to index
  %1 = arith.index_cast %0 : i32 to index
  return
}

// -----

// CHECK-LABEL: @constantLoadF32
func.func @constantLoadF32() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:f32>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:f32> -> tensor<f32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][] : tensor<f32>
  %0 = hal.interface.constant.load[0] : f32
  // CHECK: = math.absf %[[EXTRACT]] : f32
  %1 = math.absf %0 : f32
  return
}

// -----

// CHECK-LABEL: @constantLoadWithIndexAndAlignment
func.func @constantLoadWithIndexAndAlignment() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c5) alignment(16) : !flow.dispatch.tensor<readonly:index>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:index> -> tensor<index>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][] : tensor<index>
  %0 = hal.interface.constant.load[5] alignment(16) : index
  // CHECK: = arith.index_cast %[[EXTRACT]] : index to i32
  %1 = arith.index_cast %0 : index to i32
  return
}
