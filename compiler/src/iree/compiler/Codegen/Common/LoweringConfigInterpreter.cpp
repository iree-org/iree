// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LOWERINGCONFIGINTERPRETERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class LoweringConfigInterpreterPass final
    : public impl::LoweringConfigInterpreterPassBase<
          LoweringConfigInterpreterPass> {
public:
  using impl::LoweringConfigInterpreterPassBase<
      LoweringConfigInterpreterPass>::LoweringConfigInterpreterPassBase;
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    // Supports both inline strategy IR and externally cached using the
    // transform library module mechanism. Inline strategies take precedence
    // over external ones in case a symbol matches in both.
    auto *symbolTableOp = SymbolTable::getNearestSymbolTable(rootOp);
    MLIRContext *ctx = &getContext();
    auto dialect = ctx->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
    std::optional<ModuleOp> originalSpec = std::nullopt;
    bool lookedForSpec = false;

    // Collect the list of operation + strategy pairs.
    SmallVector<std::pair<Operation *, transform::NamedSequenceOp>>
        targetStrategyPairs;
    rootOp->walk([&](Operation *op) {
      IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
          getLoweringConfig(op);
      if (!loweringConfig) {
        return;
      }

      std::optional<StringRef> maybeSymName =
          loweringConfig.getLoweringStrategy();
      if (!maybeSymName) {
        return;
      }

      auto strategy = dyn_cast_or_null<transform::NamedSequenceOp>(
          SymbolTable::lookupSymbolIn(symbolTableOp, *maybeSymName));
      if (!strategy) {
        // Since potentially every executable will end up running this pass,
        // acquiring a mutex for every single one isn't great, especially since
        // the vast majority of cases will never hit this path. So only lookup
        // the module when needed.
        if (!lookedForSpec) {
          lookedForSpec = true;
          originalSpec = dialect->getLoneTransformLibraryModule();
        }
        if (originalSpec) {
          strategy = dyn_cast_or_null<transform::NamedSequenceOp>(
              SymbolTable::lookupSymbolIn(originalSpec.value(), *maybeSymName));
        }
      }

      if (!strategy) {
        return;
      }

      targetStrategyPairs.push_back(std::make_pair(op, strategy));
    });

    // Apply the lowering strategies in no particular order. It is up to the
    // underlying strategies to make sure they don't step on each others toes
    // if multiple are present.
    transform::TransformOptions options;
    options.enableExpensiveChecks(true);
    for (auto [target, strategy] : targetStrategyPairs) {
      if (failed(transform::applyTransformNamedSequence(
              target, strategy, /*transformModule=*/nullptr, options))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
