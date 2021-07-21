// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class TranslateExecutableVariantsPass
    : public PassWrapper<TranslateExecutableVariantsPass,
                         OperationPass<IREE::HAL::ExecutableVariantOp>> {
 public:
  explicit TranslateExecutableVariantsPass() {
    for (auto &targetBackend :
         getTargetBackends(getRegisteredTargetBackends())) {
      auto pm = std::make_unique<OpPassManager>(
          IREE::HAL::ExecutableVariantOp::getOperationName(),
          OpPassManager::Nesting::Implicit);
      targetBackend->buildTranslationPassPipeline(*pm);
      pipelines_.push_back({std::move(targetBackend), std::move(pm)});
    }
  }

  TranslateExecutableVariantsPass(const TranslateExecutableVariantsPass &other)
      : TranslateExecutableVariantsPass() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    for (auto &pipeline : pipelines_) {
      pipeline.passManager->getDependentDialects(registry);
    }
  }

  StringRef getArgument() const override {
    return "iree-hal-translate-executable-variants";
  }

  StringRef getDescription() const override {
    return "Translates hal.executable.variant via the target backend pipelines";
  }

  void runOnOperation() override {
    auto variantOp = getOperation();
    for (auto &pipeline : pipelines_) {
      if (variantOp.target() != pipeline.targetBackend->name()) {
        continue;
      }
      if (failed(runPipeline(*pipeline.passManager, variantOp))) {
        variantOp.emitError() << "failed to run translation of source "
                                 "executable to target executable for backend "
                              << variantOp.target();
        return signalPassFailure();
      }
      break;  // Converted successfully; break out of loop.
    }
  }

 private:
  struct Pipeline {
    std::unique_ptr<TargetBackend> targetBackend;
    std::unique_ptr<OpPassManager> passManager;
  };
  llvm::SmallVector<Pipeline, 4> pipelines_;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createTranslateExecutableVariantsPass() {
  return std::make_unique<TranslateExecutableVariantsPass>();
}

static PassRegistration<TranslateExecutableVariantsPass> pass([] {
  return std::make_unique<TranslateExecutableVariantsPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
