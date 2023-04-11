// RUN: iree-opt --split-input-file --iree-wgsl-replace-push-constants %s | FileCheck %s

// CHECK-LABEL: @emptyFunctionNoOp
func.func @emptyFunctionNoOp() {
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @constantLoadIndex
func.func @constantLoadIndex() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(uniform_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1xvector<4xi32>>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xvector<4xi32>>> -> tensor<1xvector<4xi32>>
  // CHECK: %[[TENSOR_EXTRACT:.+]] = tensor.extract %[[LOAD]][%c0{{.*}}] : tensor<1xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT:.+]] = vector.extractelement %[[TENSOR_EXTRACT]][%c0{{.*}}] : vector<4xi32>
  // CHECK: %[[CAST:.+]] = arith.index_cast %[[VECTOR_EXTRACT]] : i32 to index
  %0 = hal.interface.constant.load[0] : index
  // CHECK: = arith.index_cast %[[CAST]] : index to i32
  %1 = arith.index_cast %0 : index to i32
  return
}

// -----

// CHECK-LABEL: @constantLoadI32
func.func @constantLoadI32() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(uniform_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1xvector<4xi32>>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xvector<4xi32>>> -> tensor<1xvector<4xi32>>
  // CHECK: %[[TENSOR_EXTRACT:.+]] = tensor.extract %[[LOAD]][%c0{{.*}}] : tensor<1xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT:.+]] = vector.extractelement %[[TENSOR_EXTRACT]][%c0{{.*}}] : vector<4xi32>
  %0 = hal.interface.constant.load[0] : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT]] : i32
  %1 = math.absi %0 : i32
  return
}

// -----

// CHECK-LABEL: @constantLoadI16
func.func @constantLoadI16() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(uniform_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1xvector<4xi32>>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xvector<4xi32>>> -> tensor<1xvector<4xi32>>
  // CHECK: %[[TENSOR_EXTRACT:.+]] = tensor.extract %[[LOAD]][%c0{{.*}}] : tensor<1xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT:.+]] = vector.extractelement %[[TENSOR_EXTRACT]][%c0{{.*}}] : vector<4xi32>
  // CHECK: %[[TRUNC:.+]] = arith.trunci %[[VECTOR_EXTRACT]] : i32 to i16
  %0 = hal.interface.constant.load[0] : i16
  // CHECK: = math.absi %[[TRUNC]] : i16
  %1 = math.absi %0 : i16
  return
}

// -----

// CHECK-LABEL: @constantLoadF32
func.func @constantLoadF32() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(uniform_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1xvector<4xi32>>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xvector<4xi32>>> -> tensor<1xvector<4xi32>>
  // CHECK: %[[TENSOR_EXTRACT:.+]] = tensor.extract %[[LOAD]][%c0{{.*}}] : tensor<1xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT:.+]] = vector.extractelement %[[TENSOR_EXTRACT]][%c0{{.*}}] : vector<4xi32>
  // CHECK: %[[CAST:.+]] = arith.bitcast %[[VECTOR_EXTRACT]] : i32 to f32
  %0 = hal.interface.constant.load[0] : f32
  // CHECK: = math.absf %[[CAST]] : f32
  %1 = math.absf %0 : f32
  return
}

// -----

// CHECK-LABEL: @constantLoadWithIndexAndAlignment
func.func @constantLoadWithIndexAndAlignment() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(uniform_buffer) alignment(16) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2xvector<4xi32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xvector<4xi32>>> -> tensor<2xvector<4xi32>>
  // CHECK: %[[TENSOR_EXTRACT:.+]] = tensor.extract %[[LOAD]][%c1{{.*}}] : tensor<2xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT:.+]] = vector.extractelement %[[TENSOR_EXTRACT]][%c1{{.*}}] : vector<4xi32>
  // CHECK: %[[CAST:.+]] = arith.index_cast %[[VECTOR_EXTRACT]] : i32 to index
  %0 = hal.interface.constant.load[5] alignment(16) : index
  // CHECK: = arith.index_cast %[[CAST]] : index to i32
  %1 = arith.index_cast %0 : index to i32
  return
}

// -----

// CHECK-LABEL: @constantLoadMultiple
func.func @constantLoadMultiple() {
  // CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(3) binding(0) type(uniform_buffer) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3xvector<4xi32>>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]], offsets = [0], sizes = [3], strides = [1] : !flow.dispatch.tensor<readonly:tensor<3xvector<4xi32>>> -> tensor<3xvector<4xi32>>

  // Extracting 8 i32s from tensor<3xvector<4xi32>:
  //   [0 1 2 3][4 5 6 7][8 9 10 11]
  //    ^-----------------^
  // 0-3 use the first vec4 (tensor extract 0 then vector extract 0-3)
  // 4-7 use the second vec4 (tensor extract 1 then vector extract 0-3)
  // 8 uses the third vec4 (tensor extract 2 then vector extract 0)

  // CHECK: %[[TENSOR_EXTRACT_0:.+]] = tensor.extract %[[LOAD]][%c0{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_0:.+]] = vector.extractelement %[[TENSOR_EXTRACT_0]][%c0{{.*}}] : vector<4xi32>
  %0 = hal.interface.constant.load[0] : i32
  // CHECK: %[[TENSOR_EXTRACT_1:.+]] = tensor.extract %[[LOAD]][%c0{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_1:.+]] = vector.extractelement %[[TENSOR_EXTRACT_1]][%c1{{.*}}] : vector<4xi32>
  %1 = hal.interface.constant.load[1] : i32
  // CHECK: %[[TENSOR_EXTRACT_2:.+]] = tensor.extract %[[LOAD]][%c0{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_2:.+]] = vector.extractelement %[[TENSOR_EXTRACT_2]][%c2{{.*}}] : vector<4xi32>
  %2 = hal.interface.constant.load[2] : i32
  // CHECK: %[[TENSOR_EXTRACT_3:.+]] = tensor.extract %[[LOAD]][%c0{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_3:.+]] = vector.extractelement %[[TENSOR_EXTRACT_3]][%c3{{.*}}] : vector<4xi32>
  %3 = hal.interface.constant.load[3] : i32
  // CHECK: %[[TENSOR_EXTRACT_4:.+]] = tensor.extract %[[LOAD]][%c1{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_4:.+]] = vector.extractelement %[[TENSOR_EXTRACT_4]][%c0{{.*}}] : vector<4xi32>
  %4 = hal.interface.constant.load[4] : i32
  // CHECK: %[[TENSOR_EXTRACT_5:.+]] = tensor.extract %[[LOAD]][%c1{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_5:.+]] = vector.extractelement %[[TENSOR_EXTRACT_5]][%c1{{.*}}] : vector<4xi32>
  %5 = hal.interface.constant.load[5] : i32
  // CHECK: %[[TENSOR_EXTRACT_6:.+]] = tensor.extract %[[LOAD]][%c1{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_6:.+]] = vector.extractelement %[[TENSOR_EXTRACT_6]][%c2{{.*}}] : vector<4xi32>
  %6 = hal.interface.constant.load[6] : i32
  // CHECK: %[[TENSOR_EXTRACT_7:.+]] = tensor.extract %[[LOAD]][%c1{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_7:.+]] = vector.extractelement %[[TENSOR_EXTRACT_7]][%c3{{.*}}] : vector<4xi32>
  %7 = hal.interface.constant.load[7] : i32
  // CHECK: %[[TENSOR_EXTRACT_8:.+]] = tensor.extract %[[LOAD]][%c2{{.*}}] : tensor<3xvector<4xi32>>
  // CHECK: %[[VECTOR_EXTRACT_8:.+]] = vector.extractelement %[[TENSOR_EXTRACT_8]][%c0{{.*}}] : vector<4xi32>
  %8 = hal.interface.constant.load[8] : i32

  // CHECK: = math.absi %[[VECTOR_EXTRACT_0]] : i32
  %abs_0 = math.absi %0 : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT_1]] : i32
  %abs_1 = math.absi %1 : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT_2]] : i32
  %abs_2 = math.absi %2 : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT_3]] : i32
  %abs_3 = math.absi %3 : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT_4]] : i32
  %abs_4 = math.absi %4 : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT_5]] : i32
  %abs_5 = math.absi %5 : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT_6]] : i32
  %abs_6 = math.absi %6 : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT_7]] : i32
  %abs_7 = math.absi %7 : i32
  // CHECK: = math.absi %[[VECTOR_EXTRACT_8]] : i32
  %abs_8 = math.absi %8 : i32
  return
}
