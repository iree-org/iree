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

// Tests the printing/parsing of the custom dialect ops.
// This doesn't have much meaning here as we don't define any custom printers or
// parsers but does serve as a reference for the op usage.

// RUN: custom-opt -split-input-file %s | custom-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !iree.ref<!custom.message>) {
  %c1_i32 = constant 1 : i32
  // CHECK: "custom.print"(%arg0, %c1_i32) : (!iree.ref<!custom.message>, i32) -> ()
  "custom.print"(%arg0, %c1_i32) : (!iree.ref<!custom.message>, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @reverseOp
func @reverseOp(%arg0 : !iree.ref<!custom.message>) -> !iree.ref<!custom.message> {
  // CHECK: %0 = "custom.reverse"(%arg0) : (!iree.ref<!custom.message>) -> !iree.ref<!custom.message>
  %0 = "custom.reverse"(%arg0) : (!iree.ref<!custom.message>) -> !iree.ref<!custom.message>
  return %0 : !iree.ref<!custom.message>
}

// -----

// CHECK-LABEL: @getUniqueMessageOp
func @getUniqueMessageOp() -> !iree.ref<!custom.message> {
  // CHECK: %0 = "custom.get_unique_message"() : () -> !iree.ref<!custom.message>
  %0 = "custom.get_unique_message"() : () -> !iree.ref<!custom.message>
  return %0 : !iree.ref<!custom.message>
}
