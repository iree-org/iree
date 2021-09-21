// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSES_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace iree {
class IREEDialect;
}

namespace iree_pydm {

std::unique_ptr<OperationPass<ModuleOp>> createConvertIREEPyDMToIREEPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerIREEPyDMToRTLPass();
std::unique_ptr<OperationPass<ModuleOp>> createLinkIREEPyDMRTLPass();

#define GEN_PASS_REGISTRATION
#include "iree-dialects/Dialect/IREEPyDM/Transforms/Passes.h.inc"

}  // namespace iree_pydm
}  // namespace mlir

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSES_H
