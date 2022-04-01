// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

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
    if (!optionalName.hasValue()) return;
    auto name = optionalName.getValue();
    if (!name.contains('.')) return;

    std::string sanitizedName(name);
    std::replace(sanitizedName.begin(), sanitizedName.end(), '.', '_');
    moduleOp.setName(sanitizedName);
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createSanitizeModuleNamesPass() {
  return std::make_unique<SanitizeModuleNamesPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
