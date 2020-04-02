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
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/IREEToSPIRVPass.h"

#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/IndexComputation/IndexComputationPass.h"
#include "iree/compiler/Translation/SPIRV/Passes/Passes.h"
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/IREEToSPIRV.h"
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/XLAToSPIRV.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {

namespace {
class IREEToSPIRVPass
    : public PassWrapper<IREEToSPIRVPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};
}  // namespace

/// Generates the entry function within the SPIR-V module for dispatch function.
template <typename operation_range>
static LogicalResult lowerEntryFunctions(spirv::ModuleOp spvModule,
                                         operation_range fns) {
  // Initialize the spir-v codegenerator.
  SPIRVCodegen<
      ConstantOpSPIRVLowering, ReturnOpSPIRVLowering,
      // IREE-specific ops:
      IREELoadOpSPIRVLowering, IREEStoreOpSPIRVLowering,
      // Standard dialect unary elementwise ops:
      // Standard dialect binary elementwise ops:
      SPIRVPwOpLowering<AddFOp, spirv::FAddOp>,
      SPIRVPwOpLowering<AndOp, spirv::LogicalAndOp>,
      SPIRVPwOpLowering<DivFOp, spirv::FDivOp>,
      SPIRVPwOpLowering<MulFOp, spirv::FMulOp>,
      SPIRVPwOpLowering<SubFOp, spirv::FSubOp>,
      SPIRVPwOpLowering<RemFOp, spirv::FRemOp>,
      SPIRVPwOpLowering<AddIOp, spirv::IAddOp>,
      SPIRVPwOpLowering<SignedDivIOp, spirv::SDivOp>,
      SPIRVPwOpLowering<MulIOp, spirv::IMulOp>,
      SPIRVPwOpLowering<SubIOp, spirv::ISubOp>,
      SPIRVPwOpLowering<SignedRemIOp, spirv::SRemOp>,
      SPIRVPwOpLowering<UnsignedRemIOp, spirv::SRemOp>,
      // XLA unary elementwise ops:
      SPIRVPwOpLowering<xla_hlo::AbsOp, spirv::GLSLSAbsOp, spirv::GLSLFAbsOp>,
      SPIRVPwOpLowering<xla_hlo::CeilOp, spirv::GLSLCeilOp>,
      SPIRVPwOpLowering<xla_hlo::CosOp, spirv::GLSLCosOp>,
      SPIRVPwOpLowering<xla_hlo::SinOp, spirv::GLSLSinOp>,
      SPIRVPwOpLowering<xla_hlo::ExpOp, spirv::GLSLExpOp>,
      // TODO(ravishankarm) : For now extract-elementOp is a no-op cause index
      // propagation only supports aggregates of rank 0.
      SPIRVIndexOpLowering<ExtractElementOp>,
      SPIRVPwOpLowering<xla_hlo::FloorOp, spirv::GLSLFloorOp>,
      SPIRVPwOpLowering<xla_hlo::LogOp, spirv::GLSLLogOp>,
      SPIRVPwOpLowering<xla_hlo::NegOp, spirv::FNegateOp>,
      SPIRVPwOpLowering<xla_hlo::RsqrtOp, spirv::GLSLInverseSqrtOp>,
      SPIRVPwOpLowering<xla_hlo::SignOp, spirv::GLSLSSignOp,
                        spirv::GLSLFSignOp>,
      SPIRVPwOpLowering<xla_hlo::SqrtOp, spirv::GLSLSqrtOp>,
      SPIRVPwOpLowering<xla_hlo::TanhOp, spirv::GLSLTanhOp>,
      XLAConvertOpSPIRVLowering,
      // XLA binary elementwise ops:
      SPIRVPwOpLowering<xla_hlo::AddOp, spirv::IAddOp, spirv::FAddOp>,
      SPIRVPwOpLowering<xla_hlo::AndOp, spirv::LogicalAndOp>,
      SPIRVPwOpLowering<xla_hlo::DivOp, spirv::FDivOp>,
      SPIRVPwOpLowering<xla_hlo::MaxOp, spirv::GLSLSMaxOp, spirv::GLSLFMaxOp>,
      SPIRVPwOpLowering<xla_hlo::MinOp, spirv::GLSLSMinOp, spirv::GLSLFMinOp>,
      SPIRVPwOpLowering<xla_hlo::MulOp, spirv::IMulOp, spirv::FMulOp>,
      SPIRVPwOpLowering<xla_hlo::OrOp, spirv::LogicalOrOp>,
      SPIRVPwOpLowering<xla_hlo::SubOp, spirv::ISubOp, spirv::FSubOp>,
      // XLA other ops:
      CmpIOpSPIRVLowering, CmpFOpSPIRVLowering,
      SPIRVPwOpLowering<xla_hlo::SelectOp, spirv::SelectOp>,
      SPIRVIndexOpLowering<xla_hlo::BroadcastOp>,
      SPIRVIndexOpLowering<xla_hlo::BroadcastInDimOp>,
      SPIRVIndexOpLowering<xla_hlo::CopyOp>,
      SPIRVIndexOpLowering<xla_hlo::ReshapeOp>,
      SPIRVIndexOpLowering<xla_hlo::ReverseOp>,
      SPIRVIndexOpLowering<xla_hlo::SliceOp>,
      SPIRVIndexOpLowering<xla_hlo::TransposeOp>, XLAConcatenateOpSPIRVLowering,
      XLAGatherOpSPIRVLowering, XLAPadOpSPIRVLowering>
      spirvCodegen;

  for (auto funcOp : fns) {
    if (failed(spirvCodegen.codegen(spvModule, funcOp))) {
      return failure();
    }
  }
  return success();
}

/// Converts the affine-apply ops to SPIR-V ops by adding the patterns for
/// lowering from affine to StandardOps and StandardOps To SPIRV.
static LogicalResult convertAffineApplyOps(MLIRContext *context,
                                           spirv::ModuleOp spvModule) {
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, context);
  auto targetAttr = spirv::lookupTargetEnvOrDefault(spvModule);
  SPIRVTypeConverter typeConverter(targetAttr);
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);
  ConversionTarget target(*context);
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });
  return applyFullConversion(spvModule, target, patterns);
}

void IREEToSPIRVPass::runOnOperation() {
  auto module = getOperation();
  OpBuilder builder(module.getBodyRegion());
  auto *context = &getContext();

  // Check if there are any dispatch functions.
  SmallVector<FuncOp, 1> fns;
  module.walk([&fns](FuncOp funcOp) {
    if (isDispatchFuncImpl(funcOp)) {
      // If there are not iree.store_output operations, just return as nothing
      // to do.
      auto walkResult = funcOp.walk([](IREE::StoreOutputOp op) -> WalkResult {
        return WalkResult::interrupt();
      });
      if (!walkResult.wasInterrupted()) return;

      fns.push_back(funcOp);
    }
  });
  if (fns.size() != 1) return;

  // Create a spirv.module Op.
  auto spvModule = builder.create<spirv::ModuleOp>(
      module.getLoc(), spirv::AddressingModel::Logical,
      spirv::MemoryModel::GLSL450);

  // Generate the SPIR-V entry function for the dispatch function
  if (failed(lowerEntryFunctions(spvModule, fns))) {
    return signalPassFailure();
  }

  // Legalize AffineApplyOp generated during spir-v codegen.
  if (failed(convertAffineApplyOps(context, spvModule))) {
    return signalPassFailure();
  }

  // Remove unneeded ops.
  for (auto &op :
       llvm::make_early_inc_range(module.getBody()->getOperations())) {
    if (!isa<spirv::ModuleOp>(op) && !isa<mlir::ModuleTerminatorOp>(op)) {
      op.erase();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createIREEToSPIRVPass() {
  return std::make_unique<IREEToSPIRVPass>();
}

static PassRegistration<IREEToSPIRVPass> ireeToSPIRVPassReg(
    "convert-iree-to-spirv",
    "Convert IREE dispatch functions to SPIR-V dialect");

void addIREEToSPIRVPasses(OpPassManager &conversionPassManager) {
  // TODO(laurenzo): createLegalizeToStdPass should probably be refactored
  // in terms of conversion patterns and added to above.
  conversionPassManager.addPass(xla_hlo::createLegalizeToStdPass());
  conversionPassManager.addPass(createIndexComputationPass());
  conversionPassManager.addPass(createIREEToSPIRVPass());

  OpPassManager &spirvPasses = conversionPassManager.nest<spirv::ModuleOp>();
  spirvPasses.addPass(spirv::createLowerABIAttributesPass());
  spirvPasses.addPass(createAdjustIntegerWidthPass());
  spirvPasses.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
}

}  // namespace iree_compiler
}  // namespace mlir
