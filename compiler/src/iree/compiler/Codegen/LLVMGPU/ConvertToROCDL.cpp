// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/ArithToAMDGPU/ArithToAMDGPU.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-convert-to-rocdl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTTOROCDLPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

static llvm::cl::opt<int>
    clROCMIndexingBits("iree-hip-index-bits",
                       llvm::cl::desc("Set the bit width of indices in ROCm."),
                       llvm::cl::init(64));

namespace {

// Transform gpu.barrier -> amdgpu.lds_barrier
// IREE code generation currently only ever needs to synchronize for
// LDS operations. This conversion is to make the barrier operations
// LDS specific because the gpu.barrier contains global memory
// operations as well.
struct ReplaceGPUBarrierWithLDSBarrier
    : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.replaceOpWithNewOp<amdgpu::LDSBarrierOp>(op);
    return success();
  }
};

static void populateConvertGPUToAMDGPUPatterns(RewritePatternSet &patterns) {
  patterns.add<ReplaceGPUBarrierWithLDSBarrier>(patterns.getContext());
}

} // namespace

// Function to check valid data types on the ROCm backend.
static LogicalResult validateDataTypes(Operation *op) {
  auto operandTypes = llvm::to_vector(op->getOperandTypes());
  auto resultTypes = llvm::to_vector(op->getResultTypes());
  if (llvm::any_of(llvm::concat<Type>(operandTypes, resultTypes),
                   llvm::IsaPred<Float8E4M3FNType, Float8E5M2Type>)) {
    op->emitOpError()
        << "F8E5M2 and F8E4M3FN types are not supported on "
           "the ROCm backend; try F8E5M2FNUZ or F8E4M3FNUZ instead.";
    return failure();
  }

  return success();
}

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding ROCDL equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct ConvertToROCDLPass final
    : impl::ConvertToROCDLPassBase<ConvertToROCDLPass> {
  using impl::ConvertToROCDLPassBase<
      ConvertToROCDLPass>::ConvertToROCDLPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::GPU::IREEGPUDialect, LLVM::LLVMDialect,
                    ROCDL::ROCDLDialect, amdgpu::AMDGPUDialect, gpu::GPUDialect,
                    ub::UBDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](Operation *op) {
      if (failed(validateDataTypes(op)))
        return signalPassFailure();
    });

    if (clROCMIndexingBits != 32 && clROCMIndexingBits != 64) {
      m.emitOpError() << "unsupported: ROCm index bit widths must either be "
                         "64 or 32, got "
                      << clROCMIndexingBits;
      return signalPassFailure();
    }
    bool use32BitIndices = clROCMIndexingBits == 32;

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(m.getContext(), DataLayout(m));
    options.overrideIndexBitwidth(use32BitIndices ? 32 : 64);
    LLVMTypeConverter converter(m.getContext(), options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) {
          switch (space) {
          case gpu::AddressSpace::Global:
            return 1;
          case gpu::AddressSpace::Workgroup:
            return 3;
          case gpu::AddressSpace::Private:
            return 5;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    // Run Vector -> Vector transformations ahead of conversion to LLVM.
    {
      RewritePatternSet patterns(&getContext());
      // These patterns only convert a subset of arith that target specific
      // rocdl intrinsics (e.g. fp8 conversions).
      StringRef chipset = getGPUTargetAttr(m).getArch();
      FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
      if (failed(maybeChipset)) {
        m.emitOpError() << "Invalid chipset name: " << chipset;
        return signalPassFailure();
      }
      arith::populateArithToAMDGPUConversionPatterns(
          patterns, /*convertFP8Arithmetic=*/true, /*saturateFP8Truncf=*/false,
          /*allowPackedF16Rtz=*/false, /*chipset=*/*maybeChipset);
      arith::populateCeilFloorDivExpandOpsPatterns(patterns);
      populateConvertGPUToAMDGPUPatterns(patterns);
      populateConvertSharedMemoryAllocOps(patterns);
      populateDropSharedMemoryDeallocOpPatterns(patterns);
      populateScalarizeMathOps(patterns);
      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateBubbleVectorBitCastOpPatterns(patterns);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorInterleaveLoweringPatterns(patterns);
      vector::populateVectorInterleaveToShufflePatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      // We currently always use 64 bit indices, thus ensure the bit width of
      // the mask compare is consistent.
      vector::populateVectorMaskMaterializationPatterns(
          patterns, /*force32BitVectorIndices=*/use32BitIndices);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      // TODO: doubtful that the "default" does what one want here, it is likely
      // better to use something else.
      vector::populateVectorTransposeLoweringPatterns(
          patterns, vector::VectorTransformsOptions());
      vector::populateVectorTransferLoweringPatterns(patterns);
      arith::populateExpandBFloat16Patterns(patterns);
      if (failed(applyPatternsGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("After applying in-dialect conversions\n" << m);

    {
      RewritePatternSet patterns(&getContext());
      populateGpuRewritePatterns(patterns);
      if (failed(applyPatternsGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("After applying GPU rewrite patterns\n" << m);

    {
      // Convert arith::maximumf/minimumf ops on AMD gpus since the lowering
      // is faulty for them.
      // TODO: Remove this once the lowering in LLVM is fixed
      // (https://github.com/llvm/llvm-project/issues/67815).
      RewritePatternSet patterns(&getContext());
      populateReplaceSlowMinMaxOpsPatterns(patterns);
      if (failed(applyPatternsGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("After converting maximumf/minimumf ops\n" << m);

    {
      RewritePatternSet llvmPatterns(&getContext());
      populateLowerHALInterfaceOp(llvmPatterns);
      populateLLVMConversionPatterns(&getContext(), llvmPatterns, converter);
      populateComplexToLLVMConversionPatterns(converter, llvmPatterns);
      populateMathToLLVMConversionPatterns(converter, llvmPatterns);
      iree_compiler::populateIREEResolveExtractStridedMetadataPatterns(
          llvmPatterns);
      populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
      populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
      cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
      arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
      StringRef chipset = getGPUTargetAttr(m).getArch();
      FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
      populateAMDGPUToROCDLConversionPatterns(
          converter, llvmPatterns, maybeChipset.value_or(amdgpu::Chipset()));
      vector::populateVectorRankReducingFMAPattern(llvmPatterns);
      vector::populateVectorInsertExtractStridedSliceTransforms(llvmPatterns);
      vector::populateVectorStepLoweringPatterns(llvmPatterns);
      vector::populateVectorBitCastLoweringPatterns(llvmPatterns);
      populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
      vector::populateVectorTransferLoweringPatterns(llvmPatterns,
                                                     /*maxTransferRank=*/1);
      populateGpuToROCDLConversionPatterns(converter, llvmPatterns,
                                           gpu::amd::Runtime::Unknown);
      LLVMConversionTarget target(getContext());
      populateFuncToLLVMFuncOpConversionPattern(converter, llvmPatterns);
      configureGpuToROCDLConversionLegality(target);
      ub::populateUBToLLVMConversionPatterns(converter, llvmPatterns);

      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
        signalPassFailure();
    }

    LDBG("After converting to rocdl\n" << m);
    ConvertToDynamicSharedMemory(m);

    LDBG("After converting to dynamic shared memory\n" << m);
  }
};
} // namespace mlir::iree_compiler
