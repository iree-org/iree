// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_PASSES
#define IREE_DIALECT_LINALGEXT_PASSES

include "mlir/Pass/PassBase.td"

def LinalgExtToLoops :
    Pass<"iree-linalg-ext-to-loops", "FuncOp"> {
  let summary = "Convert LinalgExt ops to loops and Linalg ops.";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createLinalgExtToLoopsPass()";
}

def TiledOpInterfaceTiling :
    Pass<"iree-linalg-ext-tile", "FuncOp"> {
  let summary = "Test pass for tiling using TiledOpInterface";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createTiledOpInterfaceTilingPass()";
}

#endif  // IREE_DIALECT_LINALGEXT_PASSES
