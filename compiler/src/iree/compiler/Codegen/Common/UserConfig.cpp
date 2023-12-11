// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/UserConfig.h"

namespace mlir::iree_compiler {

/// Propagate the configuration annotated in the incoming IR.
LogicalResult
setUserConfig(func::FuncOp entryPointFn, Operation *computeOp,
              IREE::Codegen::CompilationInfoAttr compilationInfo) {
  if (auto translationInfo = getTranslationInfo(entryPointFn)) {
    return computeOp->emitOpError(
        "multiple ops within dispatch trying to set the translation "
        "info");
  }

  auto info = compilationInfo.getTranslationInfo();
  if (failed(setTranslationInfo(entryPointFn, info)))
    return failure();

  SmallVector<int64_t> workgroupSize = compilationInfo.getWorkgroupSizeVals();
  std::optional<int64_t> subgroupSize = compilationInfo.getSubgroupSize();
  if (failed(setDispatchConfig(entryPointFn, workgroupSize, subgroupSize))) {
    return failure();
  }

  setLoweringConfig(computeOp, compilationInfo.getLoweringConfig());
  eraseCompilationInfo(computeOp);
  return success();
}

} // namespace mlir::iree_compiler
