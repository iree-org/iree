// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"

def MaterializeHomogeneousEncodings :
  Pass<"iree-global-opt-materialize-homogeneous-encodings", "mlir::ModuleOp"> {
  let summary = "Materializes logical encodings to physical encodings if there is a single device target.";
  let constructor =
  "mlir::iree_compiler::GlobalOptimization::createMaterializeHomogeneousEncodingsPass()";
}

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSES
