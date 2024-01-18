// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {

struct SanitizeModuleNamesPass
    : public SanitizeModuleNamesBase<SanitizeModuleNamesPass> {
  void runOnOperation() override {
    // MLIR identifiers must match this regex:
    //   (letter|[_]) (letter|digit|[_$.])*
    // https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords
    //
    // IREE VM modules use the `.` (period) character for namespacing, so
    // replace any occurrences of `.` with `_`.

    auto moduleOp = getOperation();
    auto optionalName = moduleOp.getName();
    if (!optionalName.has_value())
      return;
    auto name = optionalName.value();

    moduleOp.setName(sanitizeSymbolName(name));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createSanitizeModuleNamesPass() {
  return std::make_unique<SanitizeModuleNamesPass>();
}

} // namespace mlir::iree_compiler
