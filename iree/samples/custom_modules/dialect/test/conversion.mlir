// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests the (automatic) conversion from the custom dialect to the VM dialect.
// Depending on whether any manual conversion is performed this may get complex,
// such as when versioning imports or performing optimizations.

// RUN: custom-opt %s -iree-convert-flow-to-hal -iree-shape-expand-function-ranked-shape-dims -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @tensorToMessage
func @tensorToMessage(%tensor : tensor<2x4xf32>) {
  // CHECK-DAG: [[TYPE:%.+]] = vm.const.i32 50331680 : i32
  // CHECK-DAG: [[DIM0:%.+]] = vm.const.i32 2 : i32
  // CHECK-DAG: [[DIM1:%.+]] = vm.const.i32 4 : i32
  // CHECK-NEXT: [[VIEW:%.+]] = vm.call.variadic @hal.buffer_view.create(%arg0, [
  // CHECK-SAME:     [[DIM0]], [[DIM1]]
  // CHECK-SAME: ], [[TYPE]])
  // CHECK-NEXT: [[MSG:%.+]] = vm.call @custom.buffer_to_message([[VIEW]]) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!custom.message>
  %0 = "custom.tensor_to_message"(%tensor) : (tensor<2x4xf32>) -> !custom.message
  %c1 = constant 1 : i32
  // CHECK: vm.call @custom.print([[MSG]]
  "custom.print"(%0, %c1) : (!custom.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @dynamicTensorToMessage
func @dynamicTensorToMessage(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index) {
  // CHECK-DAG: [[TYPE:%.+]] = vm.const.i32 50331680 : i32
  // CHECK-NEXT: [[VIEW:%.+]] = vm.call.variadic @hal.buffer_view.create(%arg0, [
  // CHECK-SAME:     %arg1, %arg2
  // CHECK-SAME: ], [[TYPE]])
  // CHECK-NEXT: [[MSG:%.+]] = vm.call @custom.buffer_to_message([[VIEW]]) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!custom.message>
  %shape = shapex.make_ranked_shape %arg1, %arg2 : (index, index) -> !shapex.ranked_shape<[?, ?]>
  %shaped_tensor = shapex.tie_shape %arg0, %shape : tensor<?x?xf32>, !shapex.ranked_shape<[?, ?]>
  %0 = "custom.tensor_to_message"(%shaped_tensor) : (tensor<?x?xf32>) -> !custom.message
  %c1 = constant 1 : i32
  // CHECK: vm.call @custom.print([[MSG]]
  "custom.print"(%0, %c1) : (!custom.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @dynamicTensorToMessage2
func @dynamicTensorToMessage2(%arg0 : tensor<?x?xf32>, %arg1: !shapex.ranked_shape<[?, ?]> {iree.reflection = {}}) {
  // CHECK-DAG: [[TYPE:%.+]] = vm.const.i32 50331680 : i32
  // CHECK-NEXT: [[VIEW:%.+]] = vm.call.variadic @hal.buffer_view.create(%arg0,
  // CHECK-SAME: [%arg1, %arg2], [[TYPE]])
  // CHECK-NEXT: [[MSG:%.+]] = vm.call @custom.buffer_to_message([[VIEW]]) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!custom.message>
  %shaped_tensor = shapex.tie_shape %arg0, %arg1 : tensor<?x?xf32>, !shapex.ranked_shape<[?, ?]>
  %0 = "custom.tensor_to_message"(%shaped_tensor) : (tensor<?x?xf32>) -> !custom.message
  %c1 = constant 1 : i32
  // CHECK: vm.call @custom.print([[MSG]]
  "custom.print"(%0, %c1) : (!custom.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @messageToTensor
func @messageToTensor(%arg0 : !custom.message) -> tensor<2x4xf32> {
  // CHECK: [[VIEW:%.+]] = vm.call @custom.message_to_buffer(%arg0) : (!vm.ref<!custom.message>) -> !vm.ref<!hal.buffer_view>
  %0 = "custom.message_to_tensor"(%arg0) : (!custom.message) -> tensor<2x4xf32>
  // CHECK-NEXT: [[BUFFER:%.+]] = vm.call @hal.buffer_view.buffer([[VIEW]]) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  // CHECK-NEXT: vm.return [[BUFFER]]
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @messageToTensorReturnDim
func @messageToTensorReturnDim(%arg0 : !custom.message) -> index {
  %0 = "custom.message_to_tensor"(%arg0) : (!custom.message) -> tensor<?x4xf32>
  %c0 = constant 0 : index
  %1 = dim %0, %c0 : tensor<?x4xf32>
  // CHECK: [[VIEW:%.+]] = vm.call @custom.message_to_buffer(%arg0) : (!vm.ref<!custom.message>) -> !vm.ref<!hal.buffer_view>
  // CHECK: [[BUFFER:%.+]] = vm.call @hal.buffer_view.buffer([[VIEW]]) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  // CHECK: %{{.*}} = vm.const.i32.zero
  // CHECK: [[ZERO:%.+]] = vm.const.i32.zero
  // CHECK: [[DIM:%.+]] = vm.call @hal.buffer_view.dim([[VIEW]], [[ZERO]])
  // CHECK: vm.return [[DIM]]
  return %1 : index
}

// -----

// CHECK-LABEL: @messageToTensorReturnRank
func @messageToTensorReturnRank(%arg0 : !custom.message) -> index {
  %0 = "custom.message_to_tensor"(%arg0) : (!custom.message) -> tensor<*xf32>
  %1 = rank %0 : tensor<*xf32>
  // CHECK-DAG: [[VIEW:%.+]] = vm.call @custom.message_to_buffer(%arg0) : (!vm.ref<!custom.message>) -> !vm.ref<!hal.buffer_view>
  // CHECK-DAG: [[BUFFER:%.+]] = vm.call @hal.buffer_view.buffer([[VIEW]]) : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>
  // CHECK-DAG: [[RANK:%.+]] = vm.call @hal.buffer_view.rank([[VIEW]])
  // CHECK: vm.return [[RANK]]
  return %1 : index
}

// -----

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !custom.message) {
  %c1_i32 = constant 1 : i32
  // CHECK: vm.call @custom.print(%arg0, %c1) : (!vm.ref<!custom.message>, i32) -> ()
  "custom.print"(%arg0, %c1_i32) : (!custom.message, i32) -> ()
  return
}

// CHECK: vm.import @custom.print

// -----

// CHECK-LABEL: @reverseOp
func @reverseOp(%arg0 : !custom.message) -> !custom.message {
  // CHECK: %ref = vm.call @custom.reverse(%arg0) : (!vm.ref<!custom.message>) -> !vm.ref<!custom.message>
  %0 = "custom.reverse"(%arg0) : (!custom.message) -> !custom.message
  return %0 : !custom.message
}

// CHECK: vm.import @custom.reverse

// -----

// CHECK-LABEL: @getUniqueMessageOp
func @getUniqueMessageOp() -> !custom.message {
  // CHECK: %ref = vm.call @custom.get_unique_message() : () -> !vm.ref<!custom.message>
  %0 = "custom.get_unique_message"() : () -> !custom.message
  return %0 : !custom.message
}

// CHECK: vm.import @custom.get_unique_message
