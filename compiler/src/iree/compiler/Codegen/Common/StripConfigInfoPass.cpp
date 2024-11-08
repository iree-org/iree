// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"

#define DEBUG_TYPE "iree-codegen-strip-config-info"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_STRIPCONFIGINFOPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
class StripConfigInfoPass final
    : public impl::StripConfigInfoPassBase<StripConfigInfoPass> {
  using impl::StripConfigInfoPassBase<
      StripConfigInfoPass>::StripConfigInfoPassBase;

public:
  void runOnOperation() override;
};
} // namespace

void StripConfigInfoPass::runOnOperation() {
  auto funcOp = getOperation();
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (translationInfo) {
    // Erase the translation info from function if it exists.
    eraseTranslationInfo(funcOp);
  }

  funcOp->walk([&](Operation *op) {
    if (getLoweringConfig(op)) {
      // Erase the lowering configuration from root operation if it exists.
      eraseLoweringConfig(op);
    }
  });
}

} // namespace mlir::iree_compiler
