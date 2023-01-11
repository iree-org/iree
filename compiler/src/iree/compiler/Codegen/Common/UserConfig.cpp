// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/UserConfig.h"

namespace mlir {
namespace iree_compiler {

/// Propagate the configuration annotated in the incoming IR.
LogicalResult setUserConfig(
    func::FuncOp entryPointFn, Operation *computeOp,
    IREE::Codegen::CompilationInfoAttr compilationInfo) {
  if (auto translationInfo = getTranslationInfo(entryPointFn)) {
    return computeOp->emitOpError(
        "multiple ops within dispatch trying to set the translation "
        "info");
  }

  SmallVector<int64_t> workgroupSize = compilationInfo.getWorkgroupSizeVals();
  setTranslationInfo(entryPointFn, compilationInfo.getTranslationInfo(),
                     workgroupSize);

  setLoweringConfig(computeOp, compilationInfo.getLoweringConfig());
  eraseCompilationInfo(computeOp);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
