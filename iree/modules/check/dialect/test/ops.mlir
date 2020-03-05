// Copyright 2020 Google LLC
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

// Tests the printing/parsing of the Check dialect ops.

// RUN: check-opt -split-input-file %s | check-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @expect_true
func @expect_true() {
  // CHECK: [[C:%.+]] = constant
  %c = constant 1 : i32
  // CHECK: check.expect_true([[C]]) : i32
  check.expect_true(%c) : i32
  return
}

// -----

// CHECK-LABEL: @expect_false
func @expect_false() {
  // CHECK: [[C:%.+]] = constant
  %c = constant 1 : i32
  // CHECK: check.expect_false([[C]]) : i32
  check.expect_false(%c) : i32
  return
}
