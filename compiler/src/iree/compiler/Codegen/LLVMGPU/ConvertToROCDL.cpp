// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/DebugLog.h"
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-convert-to-rocdl"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTTOROCDLPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

// Lower iree_gpu.global_subgroup_barrier to just the hardware barrier
// instruction, with NO memory fences. Fences are handled separately.
//
// Based on LDSBarrierOpLowering but without the release/acquire fences.
// Target handling:
//   no split barriers and no barrier back-off (e.g. pre-gfx90a):
//     inline asm s_barrier
//   barrier back-off but no split barriers (gfx90a-gfx11): rocdl.s.barrier
//   split barriers (gfx12+): rocdl.s.barrier.signal + rocdl.s.barrier.wait
struct LowerGlobalSubgroupBarrier
    : OpRewritePattern<IREE::GPU::GlobalSubgroupBarrierOp> {
  LowerGlobalSubgroupBarrier(MLIRContext *context,
                             const ROCDL::TargetInfo &target)
      : OpRewritePattern(context), target(target) {}

  LogicalResult matchAndRewrite(IREE::GPU::GlobalSubgroupBarrierOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool hasSplitBarriers = target.has(llvm::AMDGPU::FEAT_GFX12_INSTS);
    // Inline assembly is only needed where the hardware has neither split
    // barriers nor barrier back-off (mainly early gfx9), to bypass the
    // conservative insertion of global memory waits at barriers.
    bool requiresInlineAsm =
        !hasSplitBarriers && !target.has(llvm::AMDGPU::FEAT_BACK_OFF_BARRIER);

    if (requiresInlineAsm) {
      // Use inline asm.
      auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                      LLVM::AsmDialect::AD_ATT);
      const char *asmStr = ";;;WARNING: BREAKS DEBUG WATCHES\ns_barrier";
      rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
          op, /*resultTypes=*/TypeRange(), /*operands=*/ValueRange(),
          /*asm_string=*/asmStr, /*constraints=*/"",
          /*has_side_effects=*/true,
          /*is_align_stack=*/false, LLVM::TailCallKind::None,
          /*convergent=*/true,
          /*asm_dialect=*/asmDialectAttr,
          /*operand_attrs=*/ArrayAttr());
    } else if (!hasSplitBarriers) {
      // Use rocdl.s.barrier.
      rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);
    } else {
      // Use rocdl.s.barrier.signal + rocdl.s.barrier.wait.
      ROCDL::BarrierSignalOp::create(rewriter, loc, -1);
      rewriter.replaceOpWithNewOp<ROCDL::BarrierWaitOp>(
          op, static_cast<int16_t>(-1));
    }
    return success();
  }

private:
  ROCDL::TargetInfo target;
};

static void
populateLowerGlobalSubgroupBarrierPatterns(RewritePatternSet &patterns,
                                           const ROCDL::TargetInfo &target) {
  patterns.add<LowerGlobalSubgroupBarrier>(patterns.getContext(), target);
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
struct SwapSetPrioWithMFMA : OpRewritePattern<ROCDL::SetPrioOp> {
  using Base::Base;
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
  return isa<Floats...>(type);
}

// Validates that arith.extf/truncf operations use fp8 types supported by the
// chipset. Storage in fp8 memrefs is always allowed; only conversion operations
// require hardware support or software emulation.
//
// Note: different chips take different FP8 formats but re-use the same
// instruction and intrinsic names, so we must filter out the "wrong" FP8 here.
static LogicalResult validateDataTypes(Operation *op,
                                       const ROCDL::TargetInfo &target) {
  // Only validate arith.extf and arith.truncf - these are the operations that
  // need hardware or software conversion support. Other ops (memrefs, etc.)
  // just store fp8 data and don't need special handling.
  if (!isa<arith::ExtFOp, arith::TruncFOp>(op)) {
    return success();
  }

  if (!target.hasOcpFp8()) {
    auto pred = containsAPred<Float8E5M2Type, Float8E4M3FNType>;
    if (llvm::any_of(op->getOperandTypes(), pred) ||
        llvm::any_of(op->getResultTypes(), pred)) {
      return op->emitOpError(
          "F8E5M2 and F8E4M3FN types are not supported on "
          "gfx942 (MI-300) or older chipsets; try F8E5M2FNUZ or F8E4M3FNUZ "
          "instead, or use --iree-llvmgpu-enable-small-float-emulation "
          "to enable software emulation.");
    }
  }

  if (!target.hasFnuzFp8()) {
    auto pred = containsAPred<Float8E5M2FNUZType, Float8E4M3FNUZType>;
    if (llvm::any_of(op->getOperandTypes(), pred) ||
        llvm::any_of(op->getResultTypes(), pred)) {
      return op->emitOpError(
          "F8E5M2FNUZ and F8E4M3FNUZ types are not supported on non-gfx942 "
          "(MI-300) chipsets; try F8E5M2 or F8E4M3FN instead, or use "
          "--iree-llvmgpu-enable-small-float-emulation "
          "to enable software emulation.");
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
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::GPU::IREEGPUDialect, LLVM::LLVMDialect,
                    ROCDL::ROCDLDialect, amdgpu::AMDGPUDialect, gpu::GPUDialect,
                    ub::UBDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    IREE::GPU::TargetAttr targetAttr = getGPUTargetAttr(m);
    StringRef targetArch = targetAttr.getArch();
    // The wave size is part of the target description: targets that support
    // both wave32 and wave64 default to wave32 unless told otherwise. IREE
    // requires a single subgroup size per executable variant (the ROCM target
    // rejects exports that disagree), so resolve it once for the module from
    // the functions' translation info, defaulting to the target's preferred
    // size like the ROCM target does. If the functions disagree, fall back to
    // the target default; the mismatch is diagnosed by the ROCM target later.
    std::optional<int64_t> waveSize;
    for (FunctionOpInterface funcOp : m.getOps<FunctionOpInterface>()) {
      std::optional<int64_t> funcWaveSize = getSubgroupSize(funcOp);
      if (!funcWaveSize) {
        continue;
      }
      if (waveSize && *waveSize != *funcWaveSize) {
        waveSize = 0;
        break;
      }
      waveSize = funcWaveSize;
    }
    if (!waveSize) {
      waveSize = targetAttr.getPreferredSubgroupSize();
    }
    FailureOr<ROCDL::TargetInfo> maybeTargetInfo =
        ROCDL::TargetInfo::get(targetArch, static_cast<unsigned>(*waveSize),
                               [&] { return m.emitOpError(); });
    if (failed(maybeTargetInfo)) {
      return signalPassFailure();
    }

    LowerToLLVMOptions options(m.getContext(), DataLayout(m));
    options.overrideIndexBitwidth(64);
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
          case gpu::AddressSpace::Constant:
            return 4;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    // Run Vector -> Vector transformations ahead of conversion to LLVM.
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

    {
      RewritePatternSet patterns(&getContext());
      // These patterns only convert a subset of arith that target specific
      // rocdl intrinsics (e.g. fp8 conversions).
      WalkResult allTypesValid = m.walk([&](Operation *op) {
        if (failed(validateDataTypes(op, *maybeTargetInfo))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (allTypesValid.wasInterrupted()) {
        return signalPassFailure();
      }
      bool supportsScaledExtTrunc =
          !getGPUTargetAttr(m).getWgp().getScaledMma().empty();
      arith::populateArithToAMDGPUConversionPatterns(
          patterns, /*convertFP8Arithmetic=*/true, /*saturateFP8Truncf=*/false,
          /*allowPackedF16Rtz=*/false, supportsScaledExtTrunc,
          /*target=*/*maybeTargetInfo);
      arith::populateCeilFloorDivExpandOpsPatterns(patterns);
      populateSwapSetPrioWithMFMAPatterns(patterns);
      populateLowerGlobalSubgroupBarrierPatterns(patterns, *maybeTargetInfo);
      populateConvertSharedMemoryAllocOps(patterns);
      populateDropSharedMemoryDeallocOpPatterns(patterns);
      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateBubbleVectorBitCastOpPatterns(patterns);
      vector::populateVectorInterleaveLoweringPatterns(patterns);
      vector::populateVectorInterleaveToShufflePatterns(patterns);

      vector::populateVectorMaskOpLoweringPatterns(patterns);
      // Use 64-bit indices for mask materialization to match the index
      // bitwidth.
      vector::populateVectorMaskMaterializationPatterns(
          patterns, /*force32BitVectorIndices=*/false);
      if (failed(applyPatternsGreedily(m, std::move(patterns), config))) {
        return signalPassFailure();
      }

      // TODO: remove this once ArithToAMDGPU learns to take a PatternBenefit.
      RewritePatternSet fallbackSmallFloatPatterns(&getContext());
      arith::populateExpandScalingExtTruncPatterns(fallbackSmallFloatPatterns);
      arith::populateExpandF4E2M1Patterns(fallbackSmallFloatPatterns);
      arith::populateExpandF8E8M0Patterns(fallbackSmallFloatPatterns);
      if (failed(applyPatternsGreedily(m, std::move(fallbackSmallFloatPatterns),
                                       config))) {
        LDBG() << "Small float patterns failed\n" << m;
        return signalPassFailure();
      }
    }

    LDBG() << "After applying in-dialect conversions\n" << m;

    {
      RewritePatternSet patterns(&getContext());
      populateGpuRewritePatterns(patterns);
      populateGpuPromoteShuffleToAMDGPUPatterns(patterns, *maybeTargetInfo);
      if (failed(applyPatternsGreedily(m, std::move(patterns), config))) {
        return signalPassFailure();
      }
    }

    LDBG() << "After applying GPU rewrite patterns\n" << m;

    {
      // Convert arith::maximumf/minimumf ops on AMD gpus since the lowering
      // is faulty for them.
      // TODO: Remove this once the lowering in LLVM is fixed
      // (https://github.com/llvm/llvm-project/issues/67815).
      RewritePatternSet patterns(&getContext());
      populateReplaceSlowMinMaxOpsPatterns(patterns);
      if (failed(applyPatternsGreedily(m, std::move(patterns), config))) {
        return signalPassFailure();
      }
    }

    LDBG() << "After converting maximumf/minimumf ops\n" << m;

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
      populateAMDGPUToROCDLConversionPatterns(converter, llvmPatterns,
                                              *maybeTargetInfo);
      vector::populateVectorRankReducingFMAPattern(llvmPatterns);
      vector::populateVectorInsertExtractStridedSliceTransforms(llvmPatterns);
      vector::populateVectorStepLoweringPatterns(llvmPatterns);
      vector::populateVectorBitCastLoweringPatterns(llvmPatterns);
      populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
      vector::populateVectorTransferLoweringPatterns(llvmPatterns,
                                                     /*maxTransferRank=*/1);
      // We pass Runtime::HIP in order to enable gpu.printf for debugging.
      // At time of writing, that flag has no other effect.
      populateGpuToROCDLConversionPatterns(
          converter, llvmPatterns, gpu::amd::Runtime::HIP, *maybeTargetInfo);
      LLVMConversionTarget target(getContext());
      populateFuncToLLVMFuncOpConversionPattern(converter, llvmPatterns);
      configureGpuToROCDLConversionLegality(target);
      populateMathToROCDLConversionPatterns(converter, llvmPatterns,
                                            /*target=*/*maybeTargetInfo);
      ub::populateUBToLLVMConversionPatterns(converter, llvmPatterns);
      target.addLegalOp<IREE::Codegen::DispatchConfigOp,
                        IREE::Codegen::YieldOp>();
      target.markOpRecursivelyLegal<IREE::Codegen::DispatchConfigOp>();

      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns)))) {
        return signalPassFailure();
      }
    }

    LDBG() << "After converting to rocdl\n" << m;

    // 16 is the maximum relevant alignment for all AMD GPUs. Unceremoniously
    // set it to 16 as all of our allocations almost always have much greater
    // alignment than this.
    // TODO(qedawkins): Set this much earlier when we introduce the allocations.
    setSharedMemoryAlignment(m, 16);

    LDBG() << "After updating shared memory alignments\n" << m;
  }
};
} // namespace mlir::iree_compiler
