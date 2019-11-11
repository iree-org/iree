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

// RUN: iree-opt %s -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: @necessary_clone_not_removed
func @necessary_clone_not_removed() -> (memref<i32>, memref<i32>) {
  // CHECK: [[ORIG:%.+]] = iree.constant[dense<1> : tensor<i32>]
  %original = iree.constant[dense<1> : tensor<i32>] : memref<i32>
  // CHECK: [[CLONE:%.+]] = "iree_hl_interp.clone"([[ORIG]])
  %cloned = "iree_hl_interp.clone"(%original) : (memref<i32>) -> memref<i32>
  %other = iree.constant[dense<2> : tensor<i32>] : memref<i32>
  %empty = iree.constant[dense<[]> : tensor<0xi32>] : memref<0xi32>
  "iree_hl_interp.copy"(%other, %empty, %original, %empty, %empty) : (memref<i32>, memref<0xi32>, memref<i32>, memref<0xi32>, memref<0xi32>) -> ()
  // CHECK: return [[CLONE]], [[ORIG]]
  return %cloned, %original : memref<i32>, memref<i32>
}

// -----

// CHECK-LABEL: @unnecessary_clone_removed
func @unnecessary_clone_removed() -> memref<i32> {
  // CHECK: [[ORIG:%.+]] = iree.constant[dense<1> : tensor<i32>]
  %original = iree.constant[dense<1> : tensor<i32>] : memref<i32>
  %cloned = "iree_hl_interp.clone"(%original) : (memref<i32>) -> memref<i32>
  // CHECK: return [[ORIG]]
  return %cloned : memref<i32>
}
