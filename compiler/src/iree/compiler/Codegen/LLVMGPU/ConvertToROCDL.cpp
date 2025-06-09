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
#include "mlir/Conversion/MathToROCDL/MathToROCDL.h"
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

/// Hacky pattern to swap `s_setprio` operations with `amdgpu.mfma` ops.
/// This is needed for ping-pong scheduling patterns to prevent off
/// waves from interrupting the MFMA region of the high priority wave.
/// The IR is rewritten as follows:
///
/// rocdl.s.setprio {iree_gpu.swap_mfma = n}
/// amdgpu.mfma // 1
/// ...
/// amdgpu.mfma // n
/// amdgpu.mfma // n + 1
///
/// to
///
/// amdgpu.mfma // 1
/// ...
/// amdgpu.mfma // n
/// rocdl.s.setprio
/// amdgpu.mfma // n + 1
///
/// This only looks at successor mfmas within the same block and is best
/// effort.
constexpr StringLiteral kSwapName = "iree_gpu.swap_mfma";
struct SwapSetPrioWithMFMA : public OpRewritePattern<ROCDL::SetPrioOp> {
  using OpRewritePattern<ROCDL::SetPrioOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ROCDL::SetPrioOp setPrio,
                                PatternRewriter &rewriter) const override {
    if (!setPrio->hasAttr(kSwapName)) {
      return failure();
    }

    auto count = setPrio->getAttrOfType<IntegerAttr>(kSwapName);
    if (!count) {
      return failure();
    }

    // Remove the swap attribute no matter what to avoid reapplying this
    // pattern.
    rewriter.startOpModification(setPrio);
    setPrio->removeDiscardableAttr(kSwapName);

    Operation *current = setPrio->getNextNode();
    Operation *mfmaToSwap = nullptr;

    for (int64_t remainingToSwap = count.getInt();
         remainingToSwap > 0 && current; current = current->getNextNode()) {
      if (isa<mlir::amdgpu::MFMAOp>(current)) {
        --remainingToSwap;
        mfmaToSwap = current;
      }
    }
    if (mfmaToSwap) {
      rewriter.moveOpAfter(setPrio, mfmaToSwap);
    }
    rewriter.finalizeOpModification(setPrio);
    return success();
  }
};

static void populateSwapSetPrioWithMFMAPatterns(RewritePatternSet &patterns) {
  patterns.add<SwapSetPrioWithMFMA>(patterns.getContext());
}

} // namespace

template <typename... Floats>
static bool containsAPred(Type type) {
  type = getElementTypeOrSelf(type);
  return llvm::isa<Floats...>(type);
}

// Function to check valid data types on the ROCm backend.
// Note to readers: different chips take different FP8 formats but re-use the
// same instruction and intrinsic names, so we must filter out the "wrong" FP8
// here.
static LogicalResult validateDataTypes(Operation *op,
                                       const amdgpu::Chipset &chipset) {
  constexpr amdgpu::Chipset kGfx942 = amdgpu::Chipset(9, 4, 2);
  if (!amdgpu::hasOcpFp8(chipset)) {
    auto pred = containsAPred<Float8E5M2Type, Float8E4M3FNType>;
    if (llvm::any_of(op->getOperandTypes(), pred) ||
        llvm::any_of(op->getResultTypes(), pred)) {
      return op->emitOpError("F8E5M2 and F8E4M3FN types are not supported on "
                             "gfx942 (MI-300) or older chipsets; try "
                             "F8E5M2FNUZ or F8E4M3FNUZ instead.");
    }
  }

  if (chipset != kGfx942) {
    auto pred = containsAPred<Float8E5M2FNUZType, Float8E4M3FNUZType>;
    if (llvm::any_of(op->getOperandTypes(), pred) ||
        llvm::any_of(op->getResultTypes(), pred)) {
      return op->emitOpError(
          "F8E5M2FNUZ and F8E4M3FNUZ types are not supported on non-gfx942 "
          "(MI-300) chipsets; try F8E5M2 or F8E4M3FN instead.");
    }
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
      auto options =
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct);
      // These patterns only convert a subset of arith that target specific
      // rocdl intrinsics (e.g. fp8 conversions).
      StringRef chipset = getGPUTargetAttr(m).getArch();
      FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
      if (failed(maybeChipset)) {
        m.emitOpError() << "Invalid chipset name: " << chipset;
        return signalPassFailure();
      }
      WalkResult allTypesValid = m.walk([&](Operation *op) {
        if (failed(validateDataTypes(op, *maybeChipset))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (allTypesValid.wasInterrupted()) {
        return signalPassFailure();
      }

      arith::populateArithToAMDGPUConversionPatterns(
          patterns, /*convertFP8Arithmetic=*/true, /*saturateFP8Truncf=*/false,
          /*allowPackedF16Rtz=*/false, /*chipset=*/*maybeChipset);
      arith::populateCeilFloorDivExpandOpsPatterns(patterns);
      populateSwapSetPrioWithMFMAPatterns(patterns);
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
          patterns, options.vectorContractLowering);
      vector::populateVectorGatherLoweringPatterns(patterns);
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      // We currently always use 64 bit indices, thus ensure the bit width of
      // the mask compare is consistent.
      vector::populateVectorMaskMaterializationPatterns(
          patterns, /*force32BitVectorIndices=*/use32BitIndices);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      // TODO: doubtful that the "default" does what one want here, it is likely
      // better to use something else.
      vector::populateVectorTransposeLoweringPatterns(
          patterns, options.vectorTransposeLowering);
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
      populateGpuPromoteShuffleToAMDGPUPatterns(patterns);
      populateGpuSubgroupIdPatterns(patterns);
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
      StringRef targetArch = getGPUTargetAttr(m).getArch();
      amdgpu::Chipset chipset =
          amdgpu::Chipset::parse(targetArch).value_or(amdgpu::Chipset());
      populateAMDGPUToROCDLConversionPatterns(converter, llvmPatterns, chipset);
      vector::populateVectorRankReducingFMAPattern(llvmPatterns);
      vector::populateVectorInsertExtractStridedSliceTransforms(llvmPatterns);
      vector::populateVectorStepLoweringPatterns(llvmPatterns);
      vector::populateVectorBitCastLoweringPatterns(llvmPatterns);
      populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
      vector::populateVectorTransferLoweringPatterns(llvmPatterns,
                                                     /*maxTransferRank=*/1);
      populateGpuToROCDLConversionPatterns(converter, llvmPatterns,
                                           gpu::amd::Runtime::Unknown, chipset);
      LLVMConversionTarget target(getContext());
      populateFuncToLLVMFuncOpConversionPattern(converter, llvmPatterns);
      configureGpuToROCDLConversionLegality(target);
      populateMathToROCDLConversionPatterns(converter, llvmPatterns);
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
