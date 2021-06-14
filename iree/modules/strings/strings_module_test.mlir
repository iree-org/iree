// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func @print_example_func(%arg0 : i32) {
  %0 = "strings.i32_to_string"(%arg0) : (i32) -> !strings.string
  "strings.print"(%0) : (!strings.string) -> ()
  return
}

func @string_tensor_to_string(%arg0 : !strings.string_tensor) -> !strings.string attributes { iree.module.export, iree.abi.none } {
  %0 = "strings.string_tensor_to_string"(%arg0) : (!strings.string_tensor) -> (!strings.string)
  return %0 : !strings.string
}

func @to_string_tensor(%arg0 : !hal.buffer_view) -> !strings.string_tensor attributes { iree.module.export, iree.abi.none } {
  %0 = "strings.to_string_tensor"(%arg0) : (!hal.buffer_view) -> !strings.string_tensor
  return %0 : !strings.string_tensor
}

func @gather(%arg0 : !strings.string_tensor, %arg1 : !hal.buffer_view) -> !strings.string_tensor attributes { iree.module.export, iree.abi.none } {
  %0 = "strings.gather"(%arg0, %arg1) : (!strings.string_tensor, !hal.buffer_view) -> !strings.string_tensor
  return %0 : !strings.string_tensor
}

func @concat(%arg0 : !strings.string_tensor) -> !strings.string_tensor attributes { iree.module.export, iree.abi.none } {
  %0 = "strings.concat"(%arg0) : (!strings.string_tensor) -> !strings.string_tensor
  return %0 : !strings.string_tensor
}
