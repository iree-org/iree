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

// Tests the (automatic) conversion from the strings dialect to the VM dialect.

// RUN: strings-opt %s -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @i32ToStringOp
func @i32ToStringOp(%arg0 : i32) -> !iree.ref<!strings.string> {
  // CHECK: vm.call @strings.i32_to_string(%arg0) : (i32) -> !iree.ref<!strings.string>
  %0 = "strings.i32_to_string"(%arg0) : (i32) -> !iree.ref<!strings.string>
  return %0 : !iree.ref<!strings.string>
}

// CHECK: vm.import @strings.i32_to_string

// -----

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !iree.ref<!strings.string>) {
  // CHECK: vm.call @strings.print(%arg0) : (!iree.ref<!strings.string>)
  "strings.print"(%arg0) : (!iree.ref<!strings.string>) -> ()
  return
}

// CHECK: vm.import @strings.print
