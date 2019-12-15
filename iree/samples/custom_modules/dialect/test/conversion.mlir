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

// Tests the (automatic) conversion from the custom dialect to the VM dialect.
// Depending on whether any manual conversion is performed this may get complex,
// such as when versioning imports or performing optimizations.

// RUN: custom-opt %s -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !ireex.ref<!custom.message>) {
  %c1_i32 = constant 1 : i32
  // CHECK: vm.call @custom.print(%arg0, %c1) : (!ireex.ref<!custom.message>, i32) -> ()
  "custom.print"(%arg0, %c1_i32) : (!ireex.ref<!custom.message>, i32) -> ()
  return
}

// CHECK: vm.import @custom.print

// -----

// CHECK-LABEL: @reverseOp
func @reverseOp(%arg0 : !ireex.ref<!custom.message>) -> !ireex.ref<!custom.message> {
  // CHECK: %ref = vm.call @custom.reverse(%arg0) : (!ireex.ref<!custom.message>) -> !ireex.ref<!custom.message>
  %0 = "custom.reverse"(%arg0) : (!ireex.ref<!custom.message>) -> !ireex.ref<!custom.message>
  return %0 : !ireex.ref<!custom.message>
}

// CHECK: vm.import @custom.reverse

// -----

// CHECK-LABEL: @getUniqueMessageOp
func @getUniqueMessageOp() -> !ireex.ref<!custom.message> {
  // CHECK: %ref = vm.call @custom.get_unique_message() : () -> !ireex.ref<!custom.message>
  %0 = "custom.get_unique_message"() : () -> !ireex.ref<!custom.message>
  return %0 : !ireex.ref<!custom.message>
}

// CHECK: vm.import @custom.get_unique_message
