// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/Target/SPIRVCommon/SPIRVTarget.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

SPIRVTargetBackend::SPIRVTargetBackend(SPIRVCodegenOptions options)
    : spvCodeGenOptions_(std::move(options)) {}

void SPIRVTargetBackend::declareTargetOpsForEnv(
    IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp executableOp,
    spirv::TargetEnvAttr spvTargetEnv) {
  auto targetBuilder = OpBuilder::atBlockTerminator(&executableOp.getBlock());
  auto targetOp = targetBuilder.create<IREE::HAL::ExecutableTargetOp>(
      sourceOp.getLoc(), name(), filter_pattern());

  auto containerBuilder = OpBuilder::atBlockTerminator(&targetOp.getBlock());
  auto innerModuleOp = containerBuilder.create<ModuleOp>(sourceOp.getLoc());

  // Attach SPIR-V target environment to the target's ModuleOp.
  // If we had multiple target environments we would generate one target op
  // per environment, with each setting its own environment attribute.
  innerModuleOp->setAttr(spirv::getTargetEnvAttrName(), spvTargetEnv);
}

void SPIRVTargetBackend::buildTranslationPassPipeline(
    OpPassManager &passManager) {
  buildSPIRVTransformPassPipeline(passManager, spvCodeGenOptions_);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
