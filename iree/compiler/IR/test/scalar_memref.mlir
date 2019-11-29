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

// RUN: iree-opt %s -pass-pipeline='func(canonicalize)' | IreeFileCheck %s

// CHECK-LABEL: @fold_memref_to_memref
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @fold_memref_to_memref(%arg0 : memref<i32>) -> memref<i32> {
  // CHECK-NEXT: return [[ARG]]
  %0 = iree.memref_to_scalar(%arg0 : memref<i32>) : i32
  %1 = iree.scalar_to_memref(%0 : i32) : memref<i32>
  return %1 : memref<i32>
}

// CHECK-LABEL: @fold_scalar_to_scalar
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @fold_scalar_to_scalar(%arg0 : i32) -> i32 {
  // CHECK-NEXT: return [[ARG]]
  %0 = iree.scalar_to_memref(%arg0 : i32) : memref<i32>
  %1 = iree.memref_to_scalar(%0 : memref<i32>) : i32
  return %1 : i32
}
