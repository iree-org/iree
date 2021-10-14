// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests the printing/parsing of the custom dialect ops.
// This doesn't have much meaning here as we don't define any custom printers or
// parsers but does serve as a reference for the op usage.

// RUN: custom-opt -split-input-file %s | custom-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !custom.message) {
  %c1_i32 = arith.constant 1 : i32
  // CHECK: "custom.print"(%arg0, %c1_i32) : (!custom.message, i32) -> ()
  "custom.print"(%arg0, %c1_i32) : (!custom.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @reverseOp
func @reverseOp(%arg0 : !custom.message) -> !custom.message {
  // CHECK: %0 = "custom.reverse"(%arg0) : (!custom.message) -> !custom.message
  %0 = "custom.reverse"(%arg0) : (!custom.message) -> !custom.message
  return %0 : !custom.message
}

// -----

// CHECK-LABEL: @getUniqueMessageOp
func @getUniqueMessageOp() -> !custom.message {
  // CHECK: %0 = "custom.get_unique_message"() : () -> !custom.message
  %0 = "custom.get_unique_message"() : () -> !custom.message
  return %0 : !custom.message
}
