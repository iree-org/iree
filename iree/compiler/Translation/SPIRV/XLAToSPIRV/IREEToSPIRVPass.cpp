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

#include "iree/compiler/Translation/SPIRV/IndexComputation/IndexComputationPass.h"
#include "iree/compiler/Translation/SPIRV/Passes/Passes.h"
#include "iree/compiler/Translation/SPIRV/ReductionCodegen/ReductionCodegenPasses.h"
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/IREEToSPIRV.h"
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/XLAToSPIRV.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {

namespace {
class PrepareHLOOpConversionPass
    : public FunctionPass<PrepareHLOOpConversionPass> {
  void runOnFunction() override;
};
class IREEToSPIRVPass : public ModulePass<IREEToSPIRVPass> {
  void runOnModule() override;
};
}  // namespace

/// Generates the reduction apply function within the SPIR-V module.
template <typename operation_range>
static LogicalResult generateReductionApplyFns(spirv::ModuleOp spvModule,
                                               ModuleOp module,
                                               OpBuilder &builder,
                                               operation_range fns) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&spvModule.getBlock());

  SymbolTable table(module);
  llvm::StringSet<> reductionApplyFnSymRefs;
  SmallVector<Operation *, 1> clonedReductionFns;
  for (auto funcOp : fns) {
    if (!funcOp.getAttr("iree.executable.reduction")) {
      continue;
    }

    auto applyFnSymRef = funcOp.template getAttrOfType<FlatSymbolRefAttr>(
        "iree.executable.reduction.apply");
    if (reductionApplyFnSymRefs.count(applyFnSymRef.getValue())) {
      continue;
    }
    auto applyFn = table.lookup<FuncOp>(applyFnSymRef.getValue());
    if (!applyFn) {
      return emitError(funcOp.getLoc(), "unable to find fn ")
             << applyFnSymRef << " which is the apply function for "
             << funcOp.getName();
    }
    // Clone the reduction apply fns into the spirv module for legalization.
    clonedReductionFns.push_back(builder.clone(*applyFn.getOperation()));
  }

  return lowerReductionApplyFunction(spvModule.getContext(),
                                     clonedReductionFns);
}

/// Generates the entry function within the SPIR-V module for dispatch function.
template <typename operation_range>
static LogicalResult generateEntryFunction(spirv::ModuleOp spvModule,
                                           operation_range fns) {
  // Initialize the spir-v codegenerator.
  SPIRVCodegen<
      ConstantOpSPIRVLowering,
      // IREE-specific ops:
      IREELoadOpSPIRVLowering, IREEReturnOpSPIRVLowering,
      IREEStoreOpSPIRVLowering, IREEStoreReduceOpSPIRVLowering,
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
    // TODO(ravishankarm): FuncOps in executable that are not dispatch functions
    // are not lowered to SPIR-V. Fix this limitation.
    if (!funcOp.getAttr("iree.executable.export")) continue;

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
  SPIRVTypeConverter typeConverter;
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);
  ConversionTarget target(*context);
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });
  return applyFullConversion(spvModule, target, patterns);
}

/// Performs initial conversion on input HLO ops, applying default lowerings
/// to forms that the SPIR-V code generator knows how to handle.
void PrepareHLOOpConversionPass::runOnFunction() {
  ConversionTarget target(getContext());
  OwningRewritePatternList patterns;
  target.addLegalDialect<xla_hlo::XlaHloDialect>();

  // Unfuse batch norm into primitive ops.
  xla_hlo::PopulateUnfuseBatchNormPatterns(&getContext(), &patterns);
  target.addIllegalOp<xla_hlo::BatchNormInferenceOp>();

  if (failed(applyPartialConversion(getFunction(), target, patterns))) {
    return signalPassFailure();
  }
}

void IREEToSPIRVPass::runOnModule() {
  auto module = getModule();
  OpBuilder builder(module.getBodyRegion());
  auto *context = &getContext();

  // Create a spirv.module Op.
  auto spvModule = builder.create<spirv::ModuleOp>(
      module.getLoc(), spirv::AddressingModel::Logical,
      spirv::MemoryModel::GLSL450, spirv::Capability::Shader,
      spirv::Extension::SPV_KHR_storage_buffer_storage_class);

  auto fns = module.getOps<FuncOp>();

  // Generate the SPIR-V functions for any reduction apply functions.
  if (failed(generateReductionApplyFns(spvModule, module, builder, fns))) {
    return signalPassFailure();
  }

  // Generate the SPIR-V entry function for the dispatch function
  if (failed(generateEntryFunction(spvModule, fns))) {
    return signalPassFailure();
  }

  // Legalize AffineApplyOp generated during spir-v codegen.
  if (failed(convertAffineApplyOps(context, spvModule))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> createIREEToSPIRVPass() {
  return std::make_unique<IREEToSPIRVPass>();
}

static PassRegistration<IREEToSPIRVPass> ireeToSPIRVPassReg(
    "convert-iree-to-spirv",
    "Convert IREE dispatch functions to SPIR-V dialect");

void addIREEToSPIRVPasses(PassManager &conversionPassManager) {
  conversionPassManager.addPass(std::make_unique<PrepareHLOOpConversionPass>());
  // TODO(laurenzo): createLegalizeToStdPass should probably be refactored
  // in terms of conversion patterns and added to above.
  conversionPassManager.addPass(xla_hlo::createLegalizeToStdPass());
  conversionPassManager.addPass(createPrepareReductionDispatchPass());
  conversionPassManager.addPass(createIndexComputationPass());
  conversionPassManager.addPass(createIREEToSPIRVPass());

  OpPassManager &spirvPasses = conversionPassManager.nest<spirv::ModuleOp>();
  spirvPasses.addPass(spirv::createLowerABIAttributesPass());
  spirvPasses.addPass(createInlinerPass());
  spirvPasses.addPass(createAdjustIntegerWidthPass());
}

}  // namespace iree_compiler
}  // namespace mlir
