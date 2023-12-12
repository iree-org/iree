// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

namespace {

bool IsSPIRVBasedBackend(StringRef backend) {
  return backend.starts_with("vulkan") || backend.starts_with("metal") ||
         backend.starts_with("webgpu");
}

struct SPIRVTrimExecutableTargetEnvPass final
    : SPIRVTrimExecutableTargetEnvBase<SPIRVTrimExecutableTargetEnvPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    for (auto executable : moduleOp.getOps<IREE::HAL::ExecutableOp>()) {
      for (auto variant : executable.getOps<IREE::HAL::ExecutableVariantOp>()) {
        if (!IsSPIRVBasedBackend(variant.getTarget().getBackend())) {
          continue;
        }
        if (variant.getObjects().has_value()) {
          // Ignore external executable variants. We need to read spirv.module
          // ops to get the deduced minimal list of required capability and
          // extension. External source executables won't have any spirv.module
          // ops inside.
          continue;
        }

        mlir::ModuleOp innerModule = variant.getInnerModule();
        auto spirvModuleOps = innerModule.getOps<spirv::ModuleOp>();
        // The SPIR-V CodeGen flow in IREE guarantees that we have exactly one
        // spirv.module op inside the variant op.
        if (!llvm::hasSingleElement(spirvModuleOps)) {
          variant.emitOpError("should contain exactly one spirv.module op");
          return signalPassFailure();
        }

        spirv::ModuleOp spvModule = *spirvModuleOps.begin();
        std::optional<spirv::VerCapExtAttr> vceTriple =
            spvModule.getVceTriple();
        // The SPIR-V CodeGen flow in IREE also guarantees that we have deduced
        // the minimal version/capability/extension requirement.
        if (!vceTriple) {
          spvModule.emitError("should have deduced vce triple");
          return signalPassFailure();
        }

        // Replace the provided allow list to the minimal requirement deduced
        // from compilation.
        IREE::HAL::ExecutableTargetAttr providedTarget = variant.getTarget();
        auto deducedConfig = providedTarget.getConfiguration().replace(
            [&](spirv::TargetEnvAttr attr) { return vceTriple; });
        auto deducedTarget = IREE::HAL::ExecutableTargetAttr::get(
            providedTarget.getContext(), providedTarget.getBackend(),
            providedTarget.getFormat(), cast<DictionaryAttr>(deducedConfig));
        variant.setTargetAttr(deducedTarget);

        // Clean up the spirv.target_env attribute on inner module, which was
        // used to drive compilation.
        innerModule->removeAttr(spirv::getTargetEnvAttrName());
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createSPIRVTrimExecutableTargetEnvPass() {
  return std::make_unique<SPIRVTrimExecutableTargetEnvPass>();
}

} // namespace mlir::iree_compiler
