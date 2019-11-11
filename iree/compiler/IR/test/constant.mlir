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

// RUN: iree-opt %s | iree-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: @const
func @const() -> (memref<i32>, memref<i32>, memref<i32>, memref<i32>) {
  // CHECK: iree.constant[dense<1> : tensor<i32>] : memref<i32>
  %0 = iree.constant[dense<1> : tensor<i32>] : memref<i32>
  // CHECK-NEXT: iree.constant[dense<1> : tensor<i32>] : memref<i32>
  %1 = "iree.constant"() {value = dense<1> : tensor<i32>} : () -> memref<i32>
  // CHECK-NEXT: iree.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  %2 = "iree.constant"() {attr = "foo", value = dense<1> : tensor<i32>} : () -> memref<i32>
  // CHECK-NEXT: iree.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  %3 = iree.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  return %0, %1, %2, %3 : memref<i32>, memref<i32>, memref<i32>, memref<i32>
}
