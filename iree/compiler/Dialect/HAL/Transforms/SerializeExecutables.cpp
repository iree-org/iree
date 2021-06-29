// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

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

class SerializeExecutablesPass
    : public PassWrapper<SerializeExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
 public:
  explicit SerializeExecutablesPass(TargetOptions executableOptions)
      : executableOptions_(executableOptions) {}

  StringRef getArgument() const override {
    return "iree-hal-serialize-executables";
  }

  StringRef getDescription() const override {
    return "Serializes hal.executable.variant ops to hal.executable.binary ops";
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    auto variantOps = llvm::to_vector<4>(
        executableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      for (auto &targetBackend :
           matchTargetBackends({variantOp.target_backend_filter().str()})) {
        // Ask the target backend to serialize the executable. Note that it may
        // create one or more hal.executable.binary ops in the case of
        // multi-architecture binaries.
        OpBuilder executableBuilder(variantOp);
        if (failed(targetBackend->serializeExecutable(variantOp,
                                                      executableBuilder))) {
          variantOp.emitError() << "failed to serialize op to target backend "
                                << variantOp.target_backend_filter();
          return signalPassFailure();
        }
      }
      variantOp.erase();
    }
  }

 private:
  TargetOptions executableOptions_;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeExecutablesPass(TargetOptions executableOptions) {
  return std::make_unique<SerializeExecutablesPass>(executableOptions);
}

static PassRegistration<SerializeExecutablesPass> pass([] {
  auto options = getTargetOptionsFromFlags();
  return std::make_unique<SerializeExecutablesPass>(options);
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
