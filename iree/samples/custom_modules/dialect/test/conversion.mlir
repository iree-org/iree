// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests the (automatic) conversion from the custom dialect to the VM dialect.
// Depending on whether any manual conversion is performed this may get complex,
// such as when versioning imports or performing optimizations.

// RUN: custom-opt %s -iree-hal-conversion -iree-vm-conversion -split-input-file | FileCheck %s

// CHECK-LABEL: @tensorToMessage
func @tensorToMessage(%tensor : tensor<2x4xf32>) {
  // CHECK-NEXT: %[[MSG:.+]] = vm.call @custom.buffer_to_message(%arg0) {nosideeffects} : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!custom.message>
  %0 = "custom.tensor_to_message"(%tensor) : (tensor<2x4xf32>) -> !custom.message
  %c1 = arith.constant 1 : i32
  // CHECK: vm.call @custom.print(%[[MSG]]
  "custom.print"(%0, %c1) : (!custom.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @dynamicTensorToMessage
func @dynamicTensorToMessage(%arg0 : tensor<?x?xf32>) {
  // CHECK: %[[MSG:.+]] = vm.call @custom.buffer_to_message(%arg0) {nosideeffects} : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!custom.message>
  %0 = "custom.tensor_to_message"(%arg0) : (tensor<?x?xf32>) -> !custom.message
  %c1 = arith.constant 1 : i32
  // CHECK: vm.call @custom.print(%[[MSG]]
  "custom.print"(%0, %c1) : (!custom.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @dynamicTensorToMessage2
func @dynamicTensorToMessage2(%arg0 : tensor<?x?xf32>) {
  // CHECK: %[[MSG:.+]] = vm.call @custom.buffer_to_message(%arg0) {nosideeffects} : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!custom.message>
  %0 = "custom.tensor_to_message"(%arg0) : (tensor<?x?xf32>) -> !custom.message
  %c1 = arith.constant 1 : i32
  // CHECK: vm.call @custom.print(%[[MSG]]
  "custom.print"(%0, %c1) : (!custom.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @messageToTensor
func @messageToTensor(%arg0 : !custom.message) -> tensor<2x4xf32> {
  // CHECK: %[[VIEW:.+]] = vm.call @custom.message_to_buffer(%arg0) {nosideeffects} : (!vm.ref<!custom.message>) -> !vm.ref<!hal.buffer_view>
  %0 = "custom.message_to_tensor"(%arg0) : (!custom.message) -> tensor<2x4xf32>
  // CHECK: vm.return %[[VIEW]]
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @messageToTensorReturnDim
func @messageToTensorReturnDim(%arg0 : !custom.message) -> index {
  %0 = "custom.message_to_tensor"(%arg0) : (!custom.message) -> tensor<?x4xf32>
  %c0 = arith.constant 0 : index
  %1 = tensor.dim %0, %c0 : tensor<?x4xf32>
  // CHECK: %[[VIEW:.+]] = vm.call @custom.message_to_buffer(%arg0) {nosideeffects} : (!vm.ref<!custom.message>) -> !vm.ref<!hal.buffer_view>
  // CHECK: %{{.+}} = vm.const.i32.zero
  // CHECK: %[[ZERO:.+]] = vm.const.i32.zero
  // CHECK: %[[DIM:.+]] = vm.call @hal.buffer_view.dim(%[[VIEW]], %[[ZERO]])
  // CHECK: vm.return %[[DIM]]
  return %1 : index
}

// -----

// CHECK-LABEL: @messageToTensorReturnRank
func @messageToTensorReturnRank(%arg0 : !custom.message) -> index {
  %0 = "custom.message_to_tensor"(%arg0) : (!custom.message) -> tensor<*xf32>
  %1 = tensor.rank %0 : tensor<*xf32>
  // CHECK-DAG: %[[VIEW:.+]] = vm.call @custom.message_to_buffer(%arg0) {nosideeffects} : (!vm.ref<!custom.message>) -> !vm.ref<!hal.buffer_view>
  // CHECK-DAG: %[[RANK:.+]] = vm.call @hal.buffer_view.rank(%[[VIEW]])
  // CHECK: vm.return %[[RANK]]
  return %1 : index
}

// -----

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !custom.message) {
  %c1_i32 = arith.constant 1 : i32
  // CHECK: vm.call @custom.print(%arg0, %c1) : (!vm.ref<!custom.message>, i32) -> ()
  "custom.print"(%arg0, %c1_i32) : (!custom.message, i32) -> ()
  return
}

// CHECK: vm.import @custom.print

// -----

// CHECK-LABEL: @reverseOp
func @reverseOp(%arg0 : !custom.message) -> !custom.message {
  // CHECK: %ref = vm.call @custom.reverse(%arg0) {nosideeffects} : (!vm.ref<!custom.message>) -> !vm.ref<!custom.message>
  %0 = "custom.reverse"(%arg0) : (!custom.message) -> !custom.message
  return %0 : !custom.message
}

// CHECK: vm.import @custom.reverse

// -----

// CHECK: vm.import @custom.get_unique_message
// CHECK-LABEL: @getUniqueMessageOp
func @getUniqueMessageOp() -> !custom.message {
  // CHECK: %ref = vm.call @custom.get_unique_message() {nosideeffects} : () -> !vm.ref<!custom.message>
  %0 = "custom.get_unique_message"() : () -> !custom.message
  return %0 : !custom.message
}
