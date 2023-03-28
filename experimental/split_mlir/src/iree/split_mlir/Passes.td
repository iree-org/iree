// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SPLIT_MLIR_TRANSFORM_PASSES
#define IREE_SPLIT_MLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def OutlineFunctions :
    Pass<"iree-outline-functions", "mlir::ModuleOp"> {
  let summary = "Outline operations in separate function(s).";
  let description = [{
    Marked operation ranges in a function block are outlined/moved into new functions.
    In place of an outlined operations range is inserted a call to the new function.
    The resulting function is equivalent to the original.
    The ranges for outlining must be marked with the attributes
    `outline_range_first`, `outline_range_last`.

    Example:
    ```mlir
    func.func @f(%arg0: i32, %arg1: i32) -> i32 {
      %0 = arith.addi %arg0, %arg1 {outline_range_first} : i32
      %1 = arith.muli %0, %0 : i32
      %2 = arith.muli %1, %1 {outline_range_last} : i32
      %3 = arith.addi %2, %2 : i32
      return %3 : i32
    }
    ```

    The above MLIR will be transformed to:
    ```mlir
    func.func @f(%arg0: i32, %arg1: i32) -> i32 {
      %0 = call @f_outline_0_0(%arg0, %arg1) : (i32, i32) -> i32
      %1 = arith.addi %0, %0 : i32
      return %1 : i32
    }
    func.func @f_outline_0_0(%arg0: i32, %arg1: i32) -> i32 {
      %0 = arith.addi %arg0, %arg1 : i32
      %1 = arith.muli %0, %0 : i32
      %2 = arith.muli %1, %1 : i32
      return %2 : i32
    }
    ```

    The pass will fail if there is branching to other function blocks
    inside a marked operation range.
  }];
  let constructor = "mlir::iree::split_mlir::createOutlineFunctionsPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def MarkBisect : Pass<"iree-mark-bisect", "mlir::func::FuncOp"> {
  let summary = "Mark operations in function(s) for outlining with bisect strategy.";
  let description = [{
    Each function's entry block is bisected,
    such that each piece has balanced number of ops.
    The two pieces are marked with attributes `outline_range_first` and
    `outline_range_last`. These markings surve as input to the `OutlineFunctions` pass.

    Example:
    ```bash
    iree-opt \
      --iree-plugin=split_mlir \
      --pass-pipeline="builtin.module(func.func(iree-mark-bisect{functions=f,g}))"
      my.mlir
    ```

  }];
  let constructor = "mlir::iree::split_mlir::createMarkBisectPass()";
  let options = [
    ListOption<"functions", "functions", "std::string", "List of functions to bisect.">
  ];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif // IREE_SPLIT_MLIR_TRANSFORM_PASSES
