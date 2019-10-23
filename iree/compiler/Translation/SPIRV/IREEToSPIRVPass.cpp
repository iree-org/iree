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
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/StandardOps/Ops.h"

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
  IndexPropagationList<IndexPropagationOp<ConstantOp>,
                       // IREE-specific ops:
                       IndexPropagationOp<IREE::ReturnOp>,
                       IREELoadIndexPropagation, IREEStoreIndexPropagation,
                       // Standard dialect unary elementwise ops:
                       NoBroadcastPwOpIndexPropagation<SIToFPOp>,
                       NoBroadcastPwOpIndexPropagation<SignExtendIOp>,
                       // Standard dialect binary elementwise ops:
                       NoBroadcastPwOpIndexPropagation<AddFOp>,
                       NoBroadcastPwOpIndexPropagation<AddIOp>,
                       NoBroadcastPwOpIndexPropagation<AndOp>,
                       NoBroadcastPwOpIndexPropagation<CmpFOp>,
                       NoBroadcastPwOpIndexPropagation<CmpIOp>,
                       NoBroadcastPwOpIndexPropagation<DivFOp>,
                       NoBroadcastPwOpIndexPropagation<DivISOp>,
                       NoBroadcastPwOpIndexPropagation<DivIUOp>,
                       NoBroadcastPwOpIndexPropagation<MulFOp>,
                       NoBroadcastPwOpIndexPropagation<MulIOp>,
                       NoBroadcastPwOpIndexPropagation<OrOp>,
                       NoBroadcastPwOpIndexPropagation<RemFOp>,
                       NoBroadcastPwOpIndexPropagation<RemISOp>,
                       NoBroadcastPwOpIndexPropagation<RemIUOp>,
                       NoBroadcastPwOpIndexPropagation<SubFOp>,
                       NoBroadcastPwOpIndexPropagation<SubFOp>,
                       NoBroadcastPwOpIndexPropagation<SubIOp>,
                       NoBroadcastPwOpIndexPropagation<TruncateIOp>,
                       NoBroadcastPwOpIndexPropagation<XOrOp>,
                       NoBroadcastPwOpIndexPropagation<ZeroExtendIOp>,
                       // XLA unary elementwise ops:
                       NoBroadcastPwOpIndexPropagation<xla_hlo::AbsOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::CeilOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::ConvertOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::CosOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::ExpOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::FloorOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::LogOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::NegOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::RsqrtOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::SignOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::TanhOp>,
                       // XLA binary elementwise ops:
                       NoBroadcastPwOpIndexPropagation<xla_hlo::AddOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::AndOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::DivOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::MaxOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::MinOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::MulOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::SubOp>,
                       // XLA other ops:
                       // TODO(ravishankarm): conv, dot.
                       // TODO(ravishankarm): gather.
                       // TODO(ravishankarm): pad.
                       // TODO(ravishankarm): slice.
                       NoBroadcastPwOpIndexPropagation<xla_hlo::CopyOp>,
                       ReshapeOpIndexPropagation<xla_hlo::ReshapeOp>,
                       NoBroadcastPwOpIndexPropagation<xla_hlo::SelectOp>,
                       XLABroadcastOpIndexPropagation,
                       XLABroadcastInDimOpIndexPropagation,
                       XLAReverseOpIndexPropagation,
                       XLATransposeOpIndexPropagation>
      indexPropagation;

  // Initialize the spir-v codegenerator.
  SPIRVCodegen<
      ConstantOpSPIRVLowering,
      // IREE-specific ops:
      IREELoadOpSPIRVLowering, IREEReturnOpSPIRVLowering,
      IREEStoreOpSPIRVLowering,
      // Standard dialect unary elementwise ops:
      // Standard dialect binary elementwise ops:
      SPIRVPwOpLowering<AddFOp, spirv::FAddOp>,
      SPIRVPwOpLowering<DivFOp, spirv::FDivOp>,
      SPIRVPwOpLowering<MulFOp, spirv::FMulOp>,
      SPIRVPwOpLowering<SubFOp, spirv::FSubOp>,
      SPIRVPwOpLowering<AddIOp, spirv::IAddOp>,
      SPIRVPwOpLowering<DivISOp, spirv::SDivOp>,
      SPIRVPwOpLowering<MulIOp, spirv::IMulOp>,
      SPIRVPwOpLowering<SubIOp, spirv::ISubOp>,
      // XLA unary elementwise ops:
      SPIRVPwOpLowering<xla_hlo::AbsOp, spirv::GLSLSAbsOp, spirv::GLSLFAbsOp>,
      SPIRVPwOpLowering<xla_hlo::CeilOp, spirv::GLSLCeilOp>,
      // TODO(ravishankarm): xla_hlo::ConvertOp
      SPIRVPwOpLowering<xla_hlo::CosOp, spirv::GLSLCosOp>,
      SPIRVPwOpLowering<xla_hlo::ExpOp, spirv::GLSLExpOp>,
      SPIRVPwOpLowering<xla_hlo::FloorOp, spirv::GLSLFloorOp>,
      SPIRVPwOpLowering<xla_hlo::LogOp, spirv::GLSLLogOp>,
      SPIRVPwOpLowering<xla_hlo::NegOp, spirv::FNegateOp>,
      SPIRVPwOpLowering<xla_hlo::RsqrtOp, spirv::GLSLInverseSqrtOp>,
      SPIRVPwOpLowering<xla_hlo::SignOp, spirv::GLSLSSignOp,
                        spirv::GLSLFSignOp>,
      SPIRVPwOpLowering<xla_hlo::TanhOp, spirv::GLSLTanhOp>,
      // XLA binary elementwise ops:
      SPIRVPwOpLowering<xla_hlo::AddOp, spirv::IAddOp, spirv::FAddOp>,
      SPIRVPwOpLowering<xla_hlo::AndOp, spirv::LogicalAndOp>,
      SPIRVPwOpLowering<xla_hlo::DivOp, spirv::FDivOp>,
      SPIRVPwOpLowering<xla_hlo::MaxOp, spirv::GLSLSMaxOp, spirv::GLSLFMaxOp>,
      SPIRVPwOpLowering<xla_hlo::MinOp, spirv::GLSLSMinOp, spirv::GLSLFMinOp>,
      SPIRVPwOpLowering<xla_hlo::MulOp, spirv::IMulOp, spirv::FMulOp>,
      SPIRVPwOpLowering<xla_hlo::SubOp, spirv::ISubOp, spirv::FSubOp>,
      // XLA other ops:
      CmpFOpSPIRVLowering,
      SPIRVPwOpLowering<xla_hlo::SelectOp, spirv::SelectOp>,
      SPIRVIndexOpLowering<xla_hlo::BroadcastOp>,
      SPIRVIndexOpLowering<xla_hlo::BroadcastInDimOp>,
      SPIRVIndexOpLowering<xla_hlo::CopyOp>,
      SPIRVIndexOpLowering<xla_hlo::ReshapeOp>,
      SPIRVIndexOpLowering<xla_hlo::ReverseOp>,
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
