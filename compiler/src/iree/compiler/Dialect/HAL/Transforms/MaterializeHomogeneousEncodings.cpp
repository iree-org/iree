// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class MaterializeHomogeneousEncodingsPass
    : public PassWrapper<MaterializeHomogeneousEncodingsPass,
                         OperationPass<ModuleOp>> {
public:
  MaterializeHomogeneousEncodingsPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-materialize-homogeneous-encodings";
  }

  StringRef getDescription() const override {
    return "Mateiralizes logical encodings to physical encodings if there is "
           "a single device target.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto targetsAttr = moduleOp->getAttrOfType<ArrayAttr>("hal.device.targets");
    if (!targetsAttr || targetsAttr.size() != 1) {
      return;
    }
    auto deviceTarget = cast<IREE::HAL::DeviceTargetAttr>(targetsAttr[0]);
    SmallVector<ExecutableTargetAttr, 4> executableTargets =
        deviceTarget.getExecutableTargets();
    if (executableTargets.size() != 1) {
      return;
    }
    auto executableTarget = executableTargets[0];
    OpPassManager passManager(moduleOp.getOperationName());
    auto targetBackend =
        targetRegistry.getTargetBackend(executableTarget.getBackend());
    if (!targetBackend) {
      moduleOp.emitError() << "unregistered target backend '" << target << "'";
      return;
    }
    targetBackend->buildMaterializeEncodingsPassPipeline(executableTarget,
                                                         passManager);
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }

private:
  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<ModuleOp>>
createMaterializeHomogeneousEncodingsPass() {
  return std::make_unique<MaterializeHomogeneousEncodingsPass>();
}

static PassRegistration<MaterializeHomogeneousEncodingsPass> pass([] {
  return std::make_unique<MaterializeHomogeneousEncodingsPass>();
});

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
