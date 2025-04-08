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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LOWERINGCONFIGINTERPRETERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

constexpr StringLiteral kCodegenExternalSymbolsAttrName =
    "iree_codegen_external_symbols";

/// Look up the tuning spec in the given module or any of its parents.
static LogicalResult
getSerializedExternalSymbols(Operation *op,
                             OwningOpRef<ModuleOp> &symbolsModule) {
  auto serializedExternalModule =
      op->getAttrOfType<IREE::Util::SerializableAttrInterface>(
          kCodegenExternalSymbolsAttrName);

  if (!serializedExternalModule) {
    return success();
  }

  SmallVector<char, 0> bytecode;
  if (failed(serializedExternalModule.serializeToVector(
          op->getLoc(), llvm::endianness::native, bytecode))) {
    return op->emitError() << "Failed to read attribute "
                           << kCodegenExternalSymbolsAttrName;
  }

  ParserConfig config(serializedExternalModule.getContext());
  symbolsModule = parseSourceString<ModuleOp>(
      StringRef(bytecode.data(), bytecode.size()), config);
  if (!symbolsModule) {
    return op->emitError() << "Failed to parse module in "
                           << kCodegenExternalSymbolsAttrName;
  }
  return success();
}

namespace {
class LoweringConfigInterpreterPass final
    : public impl::LoweringConfigInterpreterPassBase<
          LoweringConfigInterpreterPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    // Supports both inline strategy IR and externally cached using the
    // transform library module mechanism. Inline strategies take precedence
    // over external ones in case a symbol matches in both.
    auto *symbolTableOp = SymbolTable::getNearestSymbolTable(rootOp);
    OwningOpRef<ModuleOp> parsedLibrary;
    if (failed(getSerializedExternalSymbols(rootOp, parsedLibrary))) {
      return signalPassFailure();
    }

    // Collect the list of operation + strategy pairs.
    SmallVector<std::pair<Operation *, transform::NamedSequenceOp>>
        targetStrategyPairs;
    WalkResult res = rootOp->walk([&](Operation *op) {
      IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
          getLoweringConfig(op);
      if (!loweringConfig) {
        return WalkResult::advance();
      }

      std::optional<StringRef> maybeSymName =
          loweringConfig.getLoweringStrategy();
      if (!maybeSymName) {
        return WalkResult::advance();
      }

      auto strategy = dyn_cast_or_null<transform::NamedSequenceOp>(
          SymbolTable::lookupSymbolIn(symbolTableOp, *maybeSymName));
      if (!strategy && parsedLibrary) {
        strategy = dyn_cast_or_null<transform::NamedSequenceOp>(
            SymbolTable::lookupSymbolIn(parsedLibrary->getOperation(),
                                        *maybeSymName));
      }

      // Fail if the strategy cannot be found for some reason. We could pass
      // through silently here as it's technically not a hard failure, however
      // this creates performance chasms on a predominantly user driven path.
      if (!strategy) {
        op->emitError("Could not find required strategy ") << *maybeSymName;
        return WalkResult::interrupt();
      }

      targetStrategyPairs.push_back({op, strategy});
      return WalkResult::advance();
    });

    if (res.wasInterrupted()) {
      return signalPassFailure();
    }

    // Apply the lowering strategies in no particular order. It is up to the
    // underlying strategies to make sure they don't step on each others toes
    // if multiple are present.
    transform::TransformOptions options;
    for (auto [target, strategy] : targetStrategyPairs) {
      if (failed(transform::applyTransformNamedSequence(
              target, strategy, /*transformModule=*/nullptr, options))) {
        return signalPassFailure();
      }
    }

    // Drop the serialized external symbols if present as we no longer need
    // them.
    if (rootOp->hasAttr(kCodegenExternalSymbolsAttrName)) {
      rootOp->removeAttr(kCodegenExternalSymbolsAttrName);
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
