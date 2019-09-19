// Copyright 2019 Google LLC
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

//===- IREEToSPIRVPass.cpp -------------------------------------*- C++//-*-===//
//
// Pass to translate iree executables for vulkan-spirv.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/IREEToSPIRVPass.h"

#include "iree/compiler/Translation/SPIRV/IREEIndexComputation.h"
#include "iree/compiler/Translation/SPIRV/IREEToSPIRV.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Dialect/SPIRV/SPIRVTypes.h"

namespace mlir {
namespace iree_compiler {

namespace {

class IREEToSPIRVPass : public ModulePass<IREEToSPIRVPass> {
  void runOnModule() override;
};

}  // namespace

void IREEToSPIRVPass::runOnModule() {
  auto module = getModule();
  OpBuilder builder(module.getBodyRegion());

  // Initialize the index computation.
  IndexPropagationList<
      IndexPropagationOp<ConstantOp>, IndexPropagationOp<IREE::ReturnOp>,
      IREELoadIndexPropagation, IREEStoreIndexPropagation,
      NoBroadcastPwOpIndexPropagation<AddFOp>,
      NoBroadcastPwOpIndexPropagation<CmpFOp>,
      NoBroadcastPwOpIndexPropagation<MulFOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::AddOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::CopyOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::ExpOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::MaxOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::MulOp>,
      ReshapeOpIndexPropagation<xla_hlo::ReshapeOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::SelectOp>,
      XLABroadcastInDimOpIndexPropagation, XLATransposeOpIndexPropagation>
      indexPropagation;

  // Initialize the spir-v codegenerator.
  SPIRVCodegen<ConstantOpSPIRVLowering, CmpFOpSPIRVLowering,
               CmpSelectOpSPIRVLowering<xla_hlo::MaxOp, spirv::SGreaterThanOp,
                                        spirv::FOrdGreaterThanOp>,
               IREELoadOpSPIRVLowering, IREEReturnOpSPIRVLowering,
               IREEStoreOpSPIRVLowering,
               SPIRVPwOpLowering<AddFOp, spirv::FAddOp>,
               SPIRVPwOpLowering<MulFOp, spirv::FMulOp>,
               SPIRVPwOpLowering<xla_hlo::AddOp, spirv::IAddOp, spirv::FAddOp>,
               SPIRVPwOpLowering<xla_hlo::MulOp, spirv::IMulOp, spirv::FMulOp>,
               SPIRVPwOpLowering<xla_hlo::ExpOp, spirv::GLSLExpOp>,
               SPIRVPwOpLowering<xla_hlo::SelectOp, spirv::SelectOp>,
               SPIRVIndexOpLowering<xla_hlo::BroadcastInDimOp>,
               SPIRVIndexOpLowering<xla_hlo::CopyOp>,
               SPIRVIndexOpLowering<xla_hlo::ReshapeOp>,
               SPIRVIndexOpLowering<xla_hlo::TransposeOp>>
      spirvCodegen;

  // Create a spirv.module Op.
  auto spvModule = builder.create<spirv::ModuleOp>(
      module.getLoc(),
      builder.getI32IntegerAttr(
          static_cast<int32_t>(spirv::AddressingModel::Logical)),
      builder.getI32IntegerAttr(
          static_cast<int32_t>(spirv::MemoryModel::GLSL450)));
  SmallVector<StringRef, 2> caps;
  caps.push_back(spirv::stringifyCapability(spirv::Capability::Shader));
  spvModule.setAttr("capabilities", builder.getStrArrayAttr(caps));
  SmallVector<StringRef, 2> exts;
  exts.push_back("SPV_KHR_storage_buffer_storage_class");
  spvModule.setAttr("extensions", builder.getStrArrayAttr(exts));

  for (auto funcOp : module.getOps<FuncOp>()) {
    // TODO(ravishankarm): FuncOps in executable that are not dispatch functions
    // are not lowered to SPIR-V. Fix this limitation.
    if (!funcOp.getAttr("iree.executable.export")) continue;

    IndexComputationCache indexMap;
    if (failed(indexPropagation.propagate(funcOp.getBody(), indexMap))) {
      return signalPassFailure();
    }
    // dumpIndexCache(indexMap);

    ValueCache valueCache;
    AffineExprCodegen affineExprCodegen(spvModule, indexMap);
    if (failed(spirvCodegen.codegen(spvModule, funcOp, affineExprCodegen,
                                    valueCache))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> createIREEToSPIRVPass() {
  return std::make_unique<IREEToSPIRVPass>();
}
static PassRegistration<IREEToSPIRVPass> pass(
    "convert-iree-to-spirv",
    "Convert IREE dispatch functions to SPIR-V dialect");

}  // namespace iree_compiler
}  // namespace mlir
