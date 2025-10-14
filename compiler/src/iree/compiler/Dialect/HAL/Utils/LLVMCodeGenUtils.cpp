// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Utils/LLVMCodeGenUtils.h"

#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir::iree_compiler::IREE::HAL {

void populateLLVMFuncTargetAttrs(ModuleOp moduleOp,
                                 const llvm::TargetMachine &targetMachine) {
  MLIRContext *context = moduleOp.getContext();
  const llvm::MCSubtargetInfo *subTargetInfo =
      targetMachine.getMCSubtargetInfo();
  StringRef targetCPU = targetMachine.getTargetCPU();

  const std::vector<llvm::SubtargetFeatureKV> enabledFeatures =
      subTargetInfo->getEnabledProcessorFeatures();
  auto plussedFeatures = llvm::map_to_vector(
      enabledFeatures, [](llvm::SubtargetFeatureKV feature) {
        return std::string("+") + feature.Key;
      });
  auto plussedFeaturesRefs = llvm::to_vector_of<StringRef>(plussedFeatures);
  auto fullTargetFeaturesAttr =
      LLVM::TargetFeaturesAttr::get(context, plussedFeaturesRefs);

  Block &bodyBlock = moduleOp.getBodyRegion().front();
  for (auto funcOp : bodyBlock.getOps<LLVM::LLVMFuncOp>()) {
    if (!funcOp.getTargetFeatures().has_value()) {
      funcOp.setTargetFeaturesAttr(fullTargetFeaturesAttr);
    }
    if (!funcOp.getTargetCpu().has_value() && !targetCPU.empty()) {
      funcOp.setTargetCpuAttr(StringAttr::get(context, targetCPU));
    }
  }
}

} // namespace mlir::iree_compiler::IREE::HAL
