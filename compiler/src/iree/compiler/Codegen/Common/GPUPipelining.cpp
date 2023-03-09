// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
//====---------------------------------------------------------------------===//
// Pass to pipeline copy to shared memory for matmul op.
//====---------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {

static const StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
static const StringLiteral kPipeliningFirstStage = "__pipelining_first_stage__";
static const StringLiteral kPipeliningExtraBarrier =
    "__pipelining_extra_barrier__";

/// Returns true if the given `memrefType` has the default numeric address space
/// 0 or a HAL descriptor type address space.
static bool hasDefaultOrHALAddressSpace(MemRefType memrefType) {
  Attribute addrSpace = memrefType.getMemorySpace();
  if (!addrSpace) return true;
  auto intAttr = addrSpace.dyn_cast<IntegerAttr>();
  // Accept both default numeric address space and HAL descriptor type address
  // space--the former is used by LLVMGPU while the latter is used by SPIR-V.
  if (intAttr && intAttr.getInt() == 0) return true;
  return addrSpace.isa<IREE::HAL::DescriptorTypeAttr>();
}

/// Returns true if the given `memrefType` has the numeric address space for
/// GPU shared memory.
static bool hasSharedMemoryAddressSpace(MemRefType memrefType) {
  auto addrSpace =
      memrefType.getMemorySpace().dyn_cast_or_null<gpu::AddressSpaceAttr>();
  return addrSpace &&
         addrSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

// Returns a new predicated operation to support unpeeled epilogue. Unpeeled
// epilogue needs to handle the last iterations within the mainloop which
// requires predicating operations, for e.g., OOB global memory access. This
// helper function predicates operations (where predication is avialable),
// checks if unpredicated operations are side-effect free and acceptable to
// execute speculatively.
static Operation* replaceOpWithPredicatedOp(Operation* op, Value pred,
                                            PatternRewriter& rewriter) {
  // Predication is only supported for AsyncCopyOp. Thus, for operations which
  // are *not* AsyncCopyOp additional checks are requrired in order to be issued
  // speculatively.
  if (!isa<nvgpu::DeviceAsyncCopyOp>(op)) {
    // Return/execute the op if it is a side effect free.
    if (mlir::isMemoryEffectFree(op)) return op;
    // Return/execute the op if it is barrier, commit group, or ldmatrix op.
    if (isa<gpu::BarrierOp, nvgpu::DeviceAsyncCreateGroupOp, nvgpu::LdMatrixOp>(
            op))
      return op;
    // Return/execute the op if it is a shared memory load.
    if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
      auto loadBaseType = loadOp.getBase().getType().cast<MemRefType>();
      if (hasSharedMemoryAddressSpace(loadBaseType)) return op;
    }
    // If we are here that means the operation does not have predication support
    // and cannot be speculatively executed. Thus, unpeeled epilogue is not
    // supported.
    assert(false &&
           "Unpeeled epilogue not supported with a side-effect instruction "
           "with no predication.");
  }

  // Replace mainloop AsyncCopy with AsyncCopy(zfill) inline asm.
  auto asyncCopyOp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op);
  auto loc = asyncCopyOp->getLoc();

  // Create srcElement Value based on the pred.
  // The next few lins generate the below code:
  // srcElement = (pred) ?  dstElements : 0;
  Value dstElements =
      rewriter.create<arith::ConstantOp>(loc, asyncCopyOp.getDstElementsAttr());
  Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto srcElements =
      rewriter.create<arith::SelectOp>(loc, pred, dstElements, c0Index);
  auto asyncCopyZfillOp = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
      loc, nvgpu::DeviceAsyncTokenType::get(asyncCopyOp.getContext()),
      asyncCopyOp.getDst(), asyncCopyOp.getDstIndices(), asyncCopyOp.getSrc(),
      asyncCopyOp.getSrcIndices(), asyncCopyOp.getDstElements(), srcElements,
      UnitAttr());

  rewriter.eraseOp(asyncCopyOp);

  // Return the newly create predicated AsyncCopyZfillOp.
  return asyncCopyZfillOp;
}

/// Helper to recursively add operation dependencies within `block` to `dep`
/// set.
static void addDepOps(llvm::SmallDenseSet<Operation*>& dep, Operation* op,
                      Block* block) {
  if (!dep.insert(op).second) return;
  for (Value operand : op->getOperands()) {
    Operation* defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == block) addDepOps(dep, defOp, block);
  }
}

/// Assign stages to the loop ops. Simple logic by default, put load from global
/// memory in stage 0 and the rest in stage 1. If store_stage = 0 then put store
/// to shared memory in stage 0 as well.
static void getPipelineStages(scf::ForOp forOp,
                              std::vector<std::pair<Operation*, unsigned>>& ops,
                              unsigned depth) {
  if (!forOp->hasAttr(kPipeliningLoopMarker)) return;

  // Track dependencies of stage 0 ops.
  llvm::SmallDenseSet<Operation*> loadDep;
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (op.hasAttr(kPipeliningFirstStage)) {
      addDepOps(loadDep, &op, forOp.getBody());
    }
  }
  // Create a modulo schedule with loads from global memory and the operations
  // it depends on in stage 0. Store to shared memory and computation are in
  // stage `maxDepth`. In order to have a correct scheduling even with back
  // edges we order stages in decreasing order.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (!loadDep.count(&op) && !isa<scf::YieldOp>(op))
      ops.push_back(std::make_pair(&op, depth));
  }
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (loadDep.count(&op)) ops.push_back(std::make_pair(&op, 0));
  }
}

static void setAsyncAnnotations(Operation* op,
                                scf::PipeliningOption::PipelinerPart part,
                                unsigned iteration, unsigned depth,
                                unsigned store_stage) {
  if (auto waitOp = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op)) {
    if (waitOp.getNumGroups()) return;
    int numGroupInFlight = 0;
    if (part == scf::PipeliningOption::PipelinerPart::Kernel) {
      numGroupInFlight = depth - 1;
    } else {
      // By construction there should be no wait op in the prologue as all the
      // wait should be in the last stage.
      assert(part == scf::PipeliningOption::PipelinerPart::Epilogue);
      // Based on the schedule we pick we know how many groups are in flight for
      // each iteration of the epilogue.
      numGroupInFlight = depth - 1 - iteration;
    }
    OpBuilder b(op);
    waitOp->setAttr(waitOp.getNumGroupsAttrName(),
                    b.getI32IntegerAttr(numGroupInFlight));
  } else if (auto barrierOp = dyn_cast<gpu::BarrierOp>(op)) {
    if (store_stage != 0 ||
        part != mlir::scf::PipeliningOption::PipelinerPart::Prologue ||
        iteration >= depth - 1)
      return;
    OpBuilder b(op);
    barrierOp->setAttr(kPipeliningExtraBarrier, b.getUnitAttr());
  }
}

/// Check if the for operations contains a shared memory copy that can be
/// pipelined and annotate operations with stage information if this is the
/// case.
static bool setPipeliningMarkers(scf::ForOp forOp, bool pipelineStoreStage) {
  bool copyToWorkgroupMemory = false;
  OpBuilder builder(forOp.getContext());
  SmallVector<Operation*> barriers;
  for (Operation& op : forOp.getBody()->getOperations()) {
    // Pipeline the most inner for op that should be a flat region.
    if (op.getNumRegions() > 0) return false;
    if (isa<gpu::BarrierOp>(op)) {
      barriers.push_back(&op);
      if (pipelineStoreStage == 0)
        op.setAttr(kPipeliningFirstStage, builder.getUnitAttr());
    }
    if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp>(op)) {
      copyToWorkgroupMemory = true;
      op.setAttr(kPipeliningFirstStage, builder.getUnitAttr());
      // async copy ops need to be moved along with previous barrier.
      for (Operation* barrier : barriers) {
        barrier->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
      }
      barriers.clear();
      continue;
    }
    auto ld = dyn_cast<vector::TransferReadOp>(op);
    if (!ld) continue;
    auto ldSrcType = ld.getSource().getType().cast<MemRefType>();
    if (!hasDefaultOrHALAddressSpace(ldSrcType) || !ld->hasOneUse()) continue;
    auto st = dyn_cast<vector::TransferWriteOp>(ld->use_begin()->getOwner());
    if (!st) continue;
    auto stSrcType = st.getSource().getType().cast<MemRefType>();
    if (!hasSharedMemoryAddressSpace(stSrcType)) continue;
    copyToWorkgroupMemory = true;
    ld->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
    if (pipelineStoreStage == 0)
      st->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
  }
  if (copyToWorkgroupMemory) {
    forOp->setAttr(kPipeliningLoopMarker, builder.getUnitAttr());
    if (pipelineStoreStage == 0 && !barriers.empty()) {
      barriers.front()->erase();
    }
  }
  return copyToWorkgroupMemory;
}

/// Apply pipeline rewrite pattern assuming the operations were already
/// annotated with stage information.
// TODO: move away from using attribute annotations.
static FailureOr<scf::ForOp> applyPipelining(scf::ForOp forOp, int64_t depth,
                                             bool epiloguePeeling,
                                             bool pipelineStoreStage) {
  scf::PipeliningOption options;
  unsigned maxDepth = depth;
  auto getSchedule = [maxDepth](
                         scf::ForOp forOp,
                         std::vector<std::pair<Operation*, unsigned>>& ops) {
    return getPipelineStages(forOp, ops, maxDepth);
  };
  auto setAnnotation = [maxDepth, pipelineStoreStage](
                           Operation* op,
                           scf::PipeliningOption::PipelinerPart part,
                           unsigned iteration) {
    return setAsyncAnnotations(op, part, iteration, maxDepth,
                               pipelineStoreStage);
  };
  options.getScheduleFn = getSchedule;
  options.annotateFn = setAnnotation;

  // Use un-peeled epilogue (i.e. epiloguePeeling=flase) only when predication
  // is avialable a.k.a. AsyncCopyOp.
  if (!epiloguePeeling) {
    options.peelEpilogue = false;
    options.predicateFn = [](Operation* op, Value pred,
                             PatternRewriter& rewriter) {
      return replaceOpWithPredicatedOp(op, pred, rewriter);
    };
  }
  scf::ForLoopPipeliningPattern pattern(options, forOp->getContext());
  transform::TrivialPatternRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return pattern.returningMatchAndRewrite(forOp, rewriter);
}
namespace {

struct MmaLdMatrixOpToNVVM : public ConvertOpToLLVMPattern<nvgpu::LdMatrixOp> {
  using ConvertOpToLLVMPattern<nvgpu::LdMatrixOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::LdMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = getContext();
    Location loc = op->getLoc();

    // The result type of ldmatrix will always be a struct of 32bit integer
    // registers if more than one 32bit value is returned. Otherwise, the result
    // is a single i32. The result type of the GPU operation is always a vector
    // of shape (NumRegisters, VectorRegister) where VectorRegister is the
    // vector type of the result and always 32 bits long. We bitcast the result
    // of the NVVM::LdMatrix to this vector type.
    auto vectorResultType = op->getResultTypes()[0].dyn_cast<VectorType>();
    if (!vectorResultType) {
      return failure();
    }
    Type innerVectorType = LLVM::getFixedVectorType(
        vectorResultType.getElementType(), vectorResultType.getDimSize(1));

    int64_t num32BitRegs = vectorResultType.getDimSize(0);

    Type ldMatrixResultType;
    if (num32BitRegs > 1) {
      ldMatrixResultType = LLVM::LLVMStructType::getLiteral(
          ctx, SmallVector<Type>(num32BitRegs, rewriter.getI32Type()));
    } else {
      ldMatrixResultType = rewriter.getI32Type();
    }

    auto srcMemrefType = op.getSrcMemref().getType().cast<MemRefType>();
    Value srcPtr = getStridedElementPtrViaAffineApply(
        loc, srcMemrefType, adaptor.getSrcMemref(),
        // HACK: Here we don't use the adaptor indices because they would come
        // with the LLVM types, whereas at this point we want to stick with the
        // higher level types that we have (index vs. iXX). Indee, we are not
        // lowering the address computation here, just making them more
        // amendable to optimization.
        op.getIndices(), rewriter);
    Value ldMatrixResult = rewriter.create<NVVM::LdMatrixOp>(
        loc, ldMatrixResultType, srcPtr,
        /*num=*/op.getNumTiles(),
        /*layout=*/op.getTranspose() ? NVVM::MMALayout::col
                                     : NVVM::MMALayout::row);

    // The ldmatrix operation returns either a single i32 value or a struct of
    // i32 values. Here we unpack those values and cast them back to their
    // actual vector type (still of width 32b) and repack them into a result
    // struct.
    Type finalResultType = typeConverter->convertType(vectorResultType);
    Value result = rewriter.create<LLVM::UndefOp>(loc, finalResultType);
    for (int64_t i = 0, e = vectorResultType.getDimSize(0); i < e; i++) {
      Value i32Register =
          num32BitRegs > 1
              ? rewriter.create<LLVM::ExtractValueOp>(loc, ldMatrixResult, i)
              : ldMatrixResult;
      Value casted =
          rewriter.create<LLVM::BitcastOp>(loc, innerVectorType, i32Register);
      result = rewriter.create<LLVM::InsertValueOp>(loc, result, casted, i);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GPUPipeliningPass : public GPUPipeliningBase<GPUPipeliningPass> {
  GPUPipeliningPass(bool epiloguePeeling, unsigned depth, unsigned storeStage)
      : depth(depth), storeStage(storeStage) {
    this->epiloguePeeling = epiloguePeeling;
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    unsigned pipelineStoreStage = storeStage;
    SmallVector<scf::ForOp> forOps;
    // Mark the loop with shared memory copy for pipelining.
    funcOp.walk([&forOps](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps) {
      if (setPipeliningMarkers(forOp, pipelineStoreStage)) {
        (void)applyPipelining(forOp, depth, epiloguePeeling,
                              pipelineStoreStage);
      }
    }
    // Remove extra barriers from the prologue assuming appropriate
    // multi-buffering.
    funcOp.walk([](gpu::BarrierOp barrierOp) {
      if (barrierOp->hasAttr(kPipeliningExtraBarrier)) barrierOp->erase();
    });
    {
      RewritePatternSet patterns(&getContext());
      LLVMTypeConverter converter(&getContext(), /*DataLayoutAnalysis=*/nullptr,
                                  /*converIndexTypeOpt=*/true);
      //    converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) ->
      //    Type {
      //      return converter.convertType(IntegerType::get(type.getContext(),
      //      32));
      //    });

      populateGpuMemorySpaceAttributeConversions(
          converter, [](gpu::AddressSpace space) -> unsigned {
            switch (space) {
              case gpu::AddressSpace::Global:
                return static_cast<unsigned>(
                    NVVM::NVVMMemorySpace::kGlobalMemorySpace);
              case gpu::AddressSpace::Workgroup:
                return static_cast<unsigned>(
                    NVVM::NVVMMemorySpace::kSharedMemorySpace);
              case gpu::AddressSpace::Private:
                return 0;
            }
            llvm_unreachable("unknown address space enum value");
            return 0;
          });
      patterns.add<MmaLdMatrixOpToNVVM>(converter);
      LLVMConversionTarget target(getContext());
      target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
      target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
      target.addLegalDialect<::mlir::AffineDialect>();
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        signalPassFailure();

    }
    {
      // At this point the ldmatrices address computations have been rewritten
      // with affine.apply.
      // Now, breakdown the affine.apply into loop variant and loop invariant
      // components.
      IRRewriter rewriter(&getContext());
      this->getOperation().walk([&](AffineApplyOp op) {
        rewriter.setInsertionPoint(op);
        reorderOperandsByHoistability(rewriter, op);
        (void)decompose(rewriter, op);
      });
    }
  }

 private:
  unsigned depth;
  unsigned storeStage;
};
}  // namespace

FailureOr<scf::ForOp> pipelineSharedMemoryCopy(
    scf::ForOp forOp, PipeliningSchedulingStrategy startegy, bool peelEpilogue,
    int64_t depth, PatternRewriter& rewriter) {
  bool pipelineStoreStage =
      startegy == PipeliningSchedulingStrategy::loadStoreStage0;
  if (setPipeliningMarkers(forOp, pipelineStoreStage)) {
    return applyPipelining(forOp, depth, peelEpilogue, pipelineStoreStage);
  }
  return failure();
}

/// Pass options
/// epiloguePeeling - try enable/disable epilogue peeling.
/// true  : Peel epilogue (no additional checks required)
/// false : Try and use unpeeled epilogue (check if predication is supported is
/// avialable)
std::unique_ptr<OperationPass<func::FuncOp>> createGPUPipeliningPass(
    bool epiloguePeeling, unsigned depth, unsigned storeStage) {
  return std::make_unique<GPUPipeliningPass>(epiloguePeeling, depth,
                                             storeStage);
}

}  // namespace iree_compiler
}  // namespace mlir
