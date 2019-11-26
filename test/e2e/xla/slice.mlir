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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="3x4xf32= 1 2 3 4 5 6 7 8 9 10 11 12" | FileCheck %s --enable-var-scope
// RUN: iree-run-mlir --target_backends=vulkan-spirv %s --input_values="3x4xf32= 1 2 3 4 5 6 7 8 9 10 11 12" | FileCheck %s --enable-var-scope

// CHECK-LABEL: EXEC @slice_whole_buffer
func @slice_whole_buffer(%arg : tensor<3x4xf32>) -> tensor<3x4xf32> {
  %result = "xla_hlo.slice"(%arg) {start_indices = dense<[0, 0]> : tensor<2xi64>, limit_indices = dense<[3, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %result : tensor<3x4xf32>
}
// CHECK: 3x4xf32=[1 2 3 4][5 6 7 8][9 10 11 12]

// CHECK-LABEL: EXEC @slice_whole_stride
func @slice_whole_stride(%arg : tensor<3x4xf32>) -> tensor<1x4xf32> {
  %result = "xla_hlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  return %result : tensor<1x4xf32>
}
// CHECK: 1x4xf32=[5 6 7 8]

// CHECK-LABEL: EXEC @slice_stride_part
func @slice_stride_part(%arg : tensor<3x4xf32>) -> tensor<1x2xf32> {
  %result = "xla_hlo.slice"(%arg) {start_indices = dense<[1, 1]> : tensor<2xi64>, limit_indices = dense<[2, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<1x2xf32>
  return %result : tensor<1x2xf32>
}
// CHECK: 1x2xf32=[6 7]

// CHECK-LABEL: EXEC @slice_multi_stride
func @slice_multi_stride(%arg : tensor<3x4xf32>) -> tensor<2x4xf32> {
  %result = "xla_hlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[3, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// CHECK: 2x4xf32=[5 6 7 8][9 10 11 12]
