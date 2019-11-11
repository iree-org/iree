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

// RUN: iree-opt %s -split-input-file -verify-diagnostics

func @tensor(%arg0 : tensor<1xf32>) {
  // expected-error@+1 {{must be memref}}
  "iree_hl_interp.tanh_f"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return
}

// -----

func @scalar(%arg0 : f32) {
  // expected-error@+1 {{must be memref}}
  "iree_hl_interp.tanh_f"(%arg0) : (f32) -> f32
  return
}

// -----

func @vector(%arg0 : vector<1xf32>) {
  // expected-error@+1 {{must be memref}}
  "iree_hl_interp.tanh_f"(%arg0) : (vector<1xf32>) -> vector<1xf32>
  return
}

// -----

func @bad_bool(%a : memref<1xf32>) {
  // expected-error@+1 {{must be memref of boolean-storing type (1 or 8 -bit integer) values}}
  "iree_hl_interp.cmp_f"(%a, %a) {predicate = 0 : i32} : (memref<1xf32>, memref<1xf32>) -> memref<1xi32>
  return
}

// -----

func @not_scalar(%a : memref<2xf32>) {
  // expected-error@+1 {{0D memref of integer values}}
  "iree_hl_interp.length"(%a) : (memref<2xf32>) -> memref<2xi32>
  return
}

// -----

func @not_scalar_int(%a : memref<1xf32>) {
  // expected-error@+1 {{0D memref of integer values}}
  "iree_hl_interp.length"(%a) : (memref<1xf32>) -> memref<f32>
  return
}

// -----

func @not_scalar_bool(%cond : memref<i32>, %a : memref<1xf32>) {
  // expected-error@+1 {{0D memref of boolean-storing type (1 or 8 -bit integer) values}}
  "iree_hl_interp.cond_assign"(%cond, %a, %a) : (memref<i32>, memref<1xf32>, memref<1xf32>) -> memref<1xf32>
  return
}

// -----

func @bad_copy(%src : memref<2xf32>, %srcIndices : memref<2xi32>, %dst : memref<2xf32>, %dstIndices : memref<2xi32>, %lengths : memref<2xi32>) {
  // expected-error@+1 {{src/dst rank is the same as srcIndices/dstIndices/lengths size}}
  "iree_hl_interp.copy"(%src, %srcIndices, %dst, %dstIndices, %lengths) : (memref<2xf32>, memref<2xi32>, memref<2xf32>, memref<2xi32>, memref<2xi32>) -> ()
  return
}
