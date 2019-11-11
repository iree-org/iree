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

func @tensor(%a: tensor<1xf32>, %dst: memref<1xf32>) {
  // expected-error@+1 {{must be memref}}
  "iree_ll_interp.tanh_f"(%a, %dst) : (tensor<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

func @scalar(%a: f32, %dst: memref<1xf32>) {
  // expected-error@+1 {{must be memref}}
  "iree_ll_interp.tanh_f"(%a, %dst) : (f32, memref<1xf32>) -> ()
  return
}

// -----

func @vector(%a: vector<1xf32>, %dst: memref<1xf32>) {
  // expected-error@+1 {{must be memref}}
  "iree_ll_interp.tanh_f"(%a, %dst) : (vector<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

func @integer_float_op(%a: memref<1xi32>, %dst: memref<1xi32>) {
  // expected-error@+1 {{must be memref of 32/64-bit float values}}
  "iree_ll_interp.tanh_f"(%a, %dst) : (memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

func @big_float(%a: memref<1xbf16>, %dst: memref<1xbf16>) {
  // expected-error@+1 {{must be memref of 32/64-bit float values}}
  "iree_ll_interp.tanh_f"(%a, %dst) : (memref<1xbf16>, memref<1xbf16>) -> ()
  return
}

// -----

func @i1_cmp(%a: memref<1xf32>, %dst: memref<1xi1>) {
  // expected-error@+1 {{must be memref of boolean-storing type (8-bit integer) values}}
  "iree_ll_interp.cmp_f"(%a, %a, %dst) {predicate = 0 : i32} : (memref<1xf32>, memref<1xf32>, memref<1xi1>) -> ()
  return
}

// -----

func @i1_cst() {
  // expected-error@+1 {{must be memref of 8/16/32/64-bit integer or 32/64-bit float values}}
  "iree_ll_interp.constant"() {value = dense<[1]> : tensor<1xi1>} : () -> memref<1xi1>
  return
}

// -----

func @i1_arg(%a : memref<1xi1>, %dst : memref<1xi1>) {
  // expected-error@+1 {{must be memref of 8/16/32/64-bit integer values}}
  "iree_ll_interp.add_i"(%a, %a, %dst) : (memref<1xi1>, memref<1xi1>, memref<1xi1>) -> ()
  return
}

// -----

func @not_int(%a : memref<1xf32>, %dst : memref<f32>) {
  // expected-error@+1 {{32-bit integer values}}
  "iree_ll_interp.length"(%a, %dst) : (memref<1xf32>, memref<f32>) -> ()
  return
}

// -----

func @wrong_int(%a : memref<1xf32>, %dst : memref<i8>) {
  // expected-error@+1 {{32-bit integer values}}
  "iree_ll_interp.length"(%a, %dst) : (memref<1xf32>, memref<i8>) -> ()
  return
}

// -----

func @not_scalar_int(%a : memref<1xf32>, %dst : memref<1xi32>) {
  // expected-error@+1 {{0D memref}}
  "iree_ll_interp.length"(%a, %dst) : (memref<1xf32>, memref<1xi32>) -> ()
  return
}

// -----

func @not_scalar_bool(%cond : memref<i32>, %a : memref<1xf32>) {
  // expected-error@+1 {{0D memref of boolean-storing type (8-bit integer) values}}
  "iree_ll_interp.cond_assign"(%cond, %a, %a) : (memref<i32>, memref<1xf32>, memref<1xf32>) -> memref<1xf32>
  return
}
