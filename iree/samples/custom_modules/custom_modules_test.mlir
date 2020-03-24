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

// Prints the %message provided reversed %count times using the native
// implementation of the "custom.print" op.
//
// See custom_modules/dialect/custom_ops.td for the op definitions and
// custom_modules/dialect/custom.imports.mlir for the import definitions.
func @reverseAndPrint(%message : !custom.message, %count : i32) -> !custom.message
    attributes { iree.module.export, iree.abi.none } {
  %c1 = constant 1 : i32
  %0 = "custom.get_unique_message"() : () -> !custom.message
  "custom.print"(%0, %c1) : (!custom.message, i32) -> ()
  %1 = call @reverse(%message) : (!custom.message) -> !custom.message
  "custom.print"(%1, %count) : (!custom.message, i32) -> ()
  return %1 : !custom.message
}

// Reverses a message. Just an example to show intra-module calls.
func @reverse(%message : !custom.message) -> !custom.message {
  %0 = "custom.reverse"(%message) : (!custom.message) -> !custom.message
  return %0 : !custom.message
}

// Prints the provided tensor to by first converting it to a message.
func @printTensor(%tensor : tensor<2x4xf32>) -> !custom.message
    attributes { iree.module.export, iree.abi.none } {
  %0 = "custom.tensor_to_message"(%tensor) : (tensor<2x4xf32>) -> !custom.message
  %c1 = constant 1 : i32
  "custom.print"(%0, %c1) : (!custom.message, i32) -> ()
  return %0 : !custom.message
}

// Round-trips a tensor through a message.
func @roundTripTensor(%tensor : tensor<2x4xf32>) -> !custom.message
    attributes { iree.module.export, iree.abi.none } {
  %0 = "custom.tensor_to_message"(%tensor) : (tensor<2x4xf32>) -> !custom.message
  %1 = "custom.message_to_tensor"(%0) : (!custom.message) -> tensor<2x4xf32>
  %2 = "custom.tensor_to_message"(%1) : (tensor<2x4xf32>) -> !custom.message
  %c1 = constant 1 : i32
  "custom.print"(%2, %c1) : (!custom.message, i32) -> ()
  return %0 : !custom.message
}
