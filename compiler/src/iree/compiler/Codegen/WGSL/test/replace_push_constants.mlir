// RUN: iree-opt --split-input-file --iree-wgsl-replace-push-constants %s | FileCheck %s

// CHECK-LABEL: @emptyFunctionNoOp
func.func @emptyFunctionNoOp() {
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @constantLoadIndex
func.func @constantLoadIndex() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1xi32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xi32>> -> tensor<1xi32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][%c0_0] : tensor<1xi32>
  // CHECK: %[[CAST:.+]] = arith.index_cast %[[EXTRACT]] : i32 to index
  %0 = hal.interface.constant.load[0] : index
  // CHECK: = arith.index_cast %[[CAST]] : index to i32
  %1 = arith.index_cast %0 : index to i32
  return
}

// -----

// CHECK-LABEL: @constantLoadI32
func.func @constantLoadI32() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1xi32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xi32>> -> tensor<1xi32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][%c0_0] : tensor<1xi32>
  %0 = hal.interface.constant.load[0] : i32
  // CHECK: = math.absi %[[EXTRACT]] : i32
  %1 = math.absi %0 : i32
  return
}

// -----

// CHECK-LABEL: @constantLoadI16
func.func @constantLoadI16() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1xi32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xi32>> -> tensor<1xi32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][%c0_0] : tensor<1xi32>
  // CHECK: %[[TRUNC:.+]] = arith.trunci %[[EXTRACT]] : i32 to i16
  %0 = hal.interface.constant.load[0] : i16
  // CHECK: = math.absi %[[TRUNC]] : i16
  %1 = math.absi %0 : i16
  return
}

// -----

// CHECK-LABEL: @constantLoadF32
func.func @constantLoadF32() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1xi32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xi32>> -> tensor<1xi32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][%c0_0] : tensor<1xi32>
  // CHECK: %[[CAST:.+]] = arith.bitcast %[[EXTRACT]] : i32 to f32
  %0 = hal.interface.constant.load[0] : f32
  // CHECK: = math.absf %[[CAST]] : f32
  %1 = math.absf %0 : f32
  return
}

// -----

// CHECK-LABEL: @constantLoadWithIndexAndAlignment
func.func @constantLoadWithIndexAndAlignment() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c5) alignment(16) : !flow.dispatch.tensor<readonly:tensor<6xi32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<6xi32>> -> tensor<6xi32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %[[LOAD]][%c5_0] : tensor<6xi32>
  // CHECK: %[[CAST:.+]] = arith.index_cast %[[EXTRACT]] : i32 to index
  %0 = hal.interface.constant.load[5] alignment(16) : index
  // CHECK: = arith.index_cast %[[CAST]] : index to i32
  %1 = arith.index_cast %0 : index to i32
  return
}

// -----

// CHECK-LABEL: @constantLoadMultiple
func.func @constantLoadMultiple() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(storage_buffer) offset(%c2) : !flow.dispatch.tensor<readonly:tensor<3xi32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [3], strides = [1] : !flow.dispatch.tensor<readonly:tensor<3xi32>> -> tensor<3xi32>

  // CHECK: %[[EXTRACT_0:.+]] = tensor.extract %[[LOAD]][%{{.*}}] : tensor<3xi32>
  %0 = hal.interface.constant.load[0] : i32
  // CHECK: %[[EXTRACT_1:.+]] = tensor.extract %[[LOAD]][%{{.*}}] : tensor<3xi32>
  %1 = hal.interface.constant.load[1] : i32
  // CHECK: %[[EXTRACT_2:.+]] = tensor.extract %[[LOAD]][%{{.*}}] : tensor<3xi32>
  %2 = hal.interface.constant.load[2] : i32

  // CHECK: = math.absi %[[EXTRACT_0]] : i32
  %3 = math.absi %0 : i32
  // CHECK: = math.absi %[[EXTRACT_1]] : i32
  %4 = math.absi %1 : i32
  // CHECK: = math.absi %[[EXTRACT_2]] : i32
  %5 = math.absi %2 : i32
  return
}
