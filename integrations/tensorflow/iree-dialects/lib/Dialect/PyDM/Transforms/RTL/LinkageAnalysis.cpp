// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/PyDM/Transforms/RTL/LinkageAnalysis.h"

#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

LinkageAnalysis::LinkageAnalysis(Operation *moduleOp) {
  moduleOp->walk<WalkOrder::PreOrder>([&](PYDM::FuncOp f) {
    if (f.empty())
      externFuncOps.push_back(f);
    // We don't need to descend into functions so just skip them.
    return WalkResult::skip();
  });
}
