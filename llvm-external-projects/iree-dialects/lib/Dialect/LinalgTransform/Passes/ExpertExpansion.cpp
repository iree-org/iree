// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "expert-expansion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]")

using namespace mlir;

/// Expands the linalg::transform::ExpertOp instances in the `module` into lists
/// of transformations as described by the `expansions` module that contains
/// PDL.
static void expandStrategyOps(ModuleOp module, ModuleOp expansions) {
  mlir::OwningOpRef<mlir::ModuleOp> clonedExpansions(
      cast<ModuleOp>(expansions->clone()));
  RewritePatternSet patterns(std::move(clonedExpansions));
  FrozenRewritePatternSet frozen(std::move(patterns));
  PatternApplicator applicator(frozen);
  applicator.applyDefaultCostModel();

  module.walk([&](linalg::transform::ExpertOp expertOp) {
    SimplePatternRewriter rewriter(expertOp);
    if (failed(applicator.matchAndRewrite(expertOp, rewriter))) {
      LLVM_DEBUG(DBGS() << "failed to rewrite strategy \""
                        << expertOp.getExpertName() << "\"\n");
    }
  });
}

namespace {
struct ExpertExpansion : public PassWrapper<ExpertExpansion, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpertExpansion)

  Pass::Option<std::string> strategyModuleName{
      *this, "strategy-module-name", llvm::cl::init("strategies"),
      llvm::cl::desc(
          "Name of the nested module containing expert strategies.")};

  explicit ExpertExpansion(StringRef name = "strategies")
      : PassWrapper<ExpertExpansion, Pass>() {
    strategyModuleName = name.str();
  }

  ExpertExpansion(const ExpertExpansion &other)
      : PassWrapper<ExpertExpansion, Pass>(other) {
    strategyModuleName = other.strategyModuleName.getValue();
  }

  StringRef getArgument() const final {
    return "linalg-transform-expert-expansion";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect>();
  }

  StringRef getDescription() const final {
    return "Expands transformation experts into individual transformations";
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module)
      return signalPassFailure();

    ModuleOp strategyModule = nullptr;
    for (auto nestedModule : module.getOps<ModuleOp>()) {
      std::optional<StringRef> name = nestedModule.getSymName();
      if (!name)
        continue;

      if (*name == strategyModuleName) {
        if (!strategyModule) {
          strategyModule = nestedModule;
          continue;
        }
        InFlightDiagnostic diag = nestedModule->emitError()
                                  << "more than one strategy module provided";
        diag.attachNote(strategyModule->getLoc()) << "previous strategy module";
        return signalPassFailure();
      }
    }

    if (!strategyModule) {
      module->emitError() << "expected a nested strategy module";
      return signalPassFailure();
    }

    expandStrategyOps(module, strategyModule);
    strategyModule->erase();
  }
};
} // namespace

void mlir::linalg::transform::registerLinalgTransformExpertExpansionPass() {
  PassRegistration<ExpertExpansion>(
      []() { return std::make_unique<ExpertExpansion>(); });
}
