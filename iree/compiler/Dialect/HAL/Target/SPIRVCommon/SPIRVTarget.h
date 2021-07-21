// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_SPIRVCOMMON_SPIRVTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_SPIRVCOMMON_SPIRVTARGET_H_

#include <string>

#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// A SPIR-V target backend that shares common overrides for Vulkan and Metal.
class SPIRVTargetBackend : public TargetBackend {
 public:
  explicit SPIRVTargetBackend(SPIRVCodegenOptions options);

  void declareVariantOpsForEnv(IREE::Flow::ExecutableOp sourceOp,
                               IREE::HAL::ExecutableOp executableOp,
                               spirv::TargetEnvAttr spvTargetEnv);

  void buildTranslationPassPipeline(OpPassManager &passManager) override;

  SPIRVCodegenOptions spvCodeGenOptions_;
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_SPIRVCOMMON_SPIRVTARGET_H_
