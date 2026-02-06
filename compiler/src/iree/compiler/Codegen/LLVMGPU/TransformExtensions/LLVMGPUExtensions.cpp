// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LLVMGPUExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using llvm::dbgs;

#define DEBUG_TYPE "transform-llvmgpu-extensions"
#define DEBUG_TYPE_ALIAS "transform-llvmgpu-extensions-alias"
#define DEBUG_VECTOR_TO_MMA "transform-llvmgpu-extensions-vector-to-mma"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBGS_ALIAS() (dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")
#define DBGS_VECTOR_TO_MMA() (dbgs() << '[' << DEBUG_VECTOR_TO_MMA << "] ")

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;
using mlir::iree_compiler::IREE::Codegen::TileSwizzle;

iree_compiler::IREE::transform_dialect::LLVMGPUExtensions::LLVMGPUExtensions() {
  // CreateAsyncGroupsOp depends on the following two dialects.
  declareGeneratedDialect<gpu::GPUDialect>();
  declareGeneratedDialect<nvgpu::NVGPUDialect>();
  declareGeneratedDialect<amdgpu::AMDGPUDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectLLVMGPUExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::LLVMGPUExtensions>();
}

//===---------------------------------------------------------------------===//
// IREE-specific LLVMGPU transformations.
//===---------------------------------------------------------------------===//

// TODO: if the number of threads was wired like the workgroup_count, we could
// reuse most of the code and not require a static number of threads.
// TODO: synchronizations for imperfectly nested stuff.
DiagnosedSilenceableFailure
transform_dialect::MapNestedForallToGpuThreadsOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto transformOp = cast<transform::TransformOpInterface>(getOperation());

  rewriter.setInsertionPointToStart(&target.getFunctionBody().front());
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(target);
  DiagnosedSilenceableFailure diag =
      mlir::transform::gpu::mapNestedForallToThreadsImpl(
          rewriter, transformOp, target, getWorkgroupDims(), getSubgroupSize(),
          getSyncAfterDistribution());
  if (!diag.succeeded()) {
    return diag;
  }

  IREE::Codegen::TranslationInfoAttr updatedTranslationInfo =
      IREE::Codegen::TranslationInfoAttr::get(
          rewriter.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::None, getWorkgroupDims(),
          getSubgroupSize());

  // Set config dictionary.
  // Transform Dialect pipeline requires translation_info pass pipeline to
  // be set to None here.
  if (translationInfo) {
    updatedTranslationInfo = IREE::Codegen::TranslationInfoAttr::get(
        rewriter.getContext(), updatedTranslationInfo.getPassPipeline(),
        updatedTranslationInfo.getCodegenSpec(),
        updatedTranslationInfo.getWorkgroupSize(),
        updatedTranslationInfo.getSubgroupSize(),
        translationInfo.getConfiguration());
  }

  if (failed(setTranslationInfo(target, updatedTranslationInfo))) {
    target->emitOpError("failed to update translation info");
    return emitDefaultDefiniteFailure(target);
  }

  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::MapNestedForallToGpuThreadsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// VectorToWarpExecuteOnLane0Op.
//===---------------------------------------------------------------------===//
void transform_dialect::VectorToWarpExecuteOnLane0Op::build(
    OpBuilder &builder, OperationState &result, Value target,
    int64_t warpSize) {
  MLIRContext *ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(
      VectorToWarpExecuteOnLane0Op::getWarpSizeAttrName(result.name),
      builder.getI64IntegerAttr(warpSize));
  result.addTypes({transform::AnyOpType::get(ctx)});
}

/// Helper method to replace all uses of the laneId operand by the constant
/// 0 inside the region. This is a necessary prerequisite to perform any kind of
/// hoisting of IR that is inside the region.
/// Return success if any replacement occurred, failure otherwise.
// TODO: this is currently brittle, what we really need here is a scope-aware
// SCCP.
static LogicalResult
replaceAllUsesOfLaneWithin(RewriterBase &b,
                           gpu::WarpExecuteOnLane0Op executeOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(executeOp);
  Value zero = arith::ConstantIndexOp::create(b, executeOp.getLoc(), 0);
  b.setInsertionPointToStart(&executeOp.getWarpRegion().front());
  Value laneId = executeOp.getLaneid();
  bool applied = false;
  for (Operation *user : llvm::make_early_inc_range(laneId.getUsers())) {
    if (!executeOp->isProperAncestor(user)) {
      continue;
    }
    b.startOpModification(user);
    user->replaceUsesOfWith(laneId, zero);
    b.finalizeOpModification(user);
    applied = true;
  }
  return success(applied);
}

/// Return the gpu::ThreadIdOp for which the predicate if equivalent to
/// `if (threadIdx.x == 0)`.
// TODO: Figure out the proper canonicalization and drop the complexity here.
// TODO: More sophisticated detection for matching
//   (threadIdx.x == 0 && other stuff not involving threadIdx.x)
static FailureOr<gpu::ThreadIdOp> isThreadIdxxZeroPredicate(scf::IfOp ifOp) {
  if (!ifOp || ifOp.getNumResults() > 0 ||
      ifOp.getThenRegion().getBlocks().size() != 1 ||
      !ifOp.getElseRegion().empty()) {
    return failure();
  }
  auto pred = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!pred) {
    return failure();
  }
  auto EQ = arith::CmpIPredicate::eq;
  auto SLT = arith::CmpIPredicate::slt;
  auto SLE = arith::CmpIPredicate::sle;
  auto ULT = arith::CmpIPredicate::ult;
  auto ULE = arith::CmpIPredicate::ule;
  if (auto threadIdOp = pred.getLhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.getDimension() != gpu::Dimension::x) {
      return failure();
    }
    if (pred.getPredicate() == EQ && isZeroInteger(pred.getRhs())) {
      return threadIdOp;
    }
    if (pred.getPredicate() == SLE && isZeroInteger(pred.getRhs())) {
      return threadIdOp;
    }
    if (pred.getPredicate() == ULE && isZeroInteger(pred.getRhs())) {
      return threadIdOp;
    }
    if (pred.getPredicate() == SLT && isOneInteger(pred.getRhs())) {
      return threadIdOp;
    }
    if (pred.getPredicate() == ULT && isOneInteger(pred.getRhs())) {
      return threadIdOp;
    }
  }
  auto SGT = arith::CmpIPredicate::sgt;
  auto SGE = arith::CmpIPredicate::sge;
  auto UGT = arith::CmpIPredicate::ugt;
  auto UGE = arith::CmpIPredicate::uge;
  if (auto threadIdOp = pred.getRhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.getDimension() != gpu::Dimension::x) {
      return failure();
    }
    if (pred.getPredicate() == EQ && isZeroInteger(pred.getLhs())) {
      return threadIdOp;
    }
    if (pred.getPredicate() == SGE && isZeroInteger(pred.getLhs())) {
      return threadIdOp;
    }
    if (pred.getPredicate() == UGE && isZeroInteger(pred.getLhs())) {
      return threadIdOp;
    }
    if (pred.getPredicate() == SGT && isOneInteger(pred.getLhs())) {
      return threadIdOp;
    }
    if (pred.getPredicate() == UGT && isOneInteger(pred.getLhs())) {
      return threadIdOp;
    }
  }
  return failure();
}

struct VectorDistributionResult {
  gpu::WarpExecuteOnLane0Op warpOp;
};

static FailureOr<VectorDistributionResult>
rewriteScfIfAsWarpExecuteOnLane0(RewriterBase &rewriter, Location loc,
                                 scf::IfOp ifOp, int64_t workgroupSizeX,
                                 int64_t warpSize) {
  // Bail if cond is not `if (threadIdx.x == 0)`.
  FailureOr<gpu::ThreadIdOp> maybeThreadIdxxOp =
      isThreadIdxxZeroPredicate(ifOp);
  if (failed(maybeThreadIdxxOp)) {
    return failure();
  }

  // All the code below will be executed on a single warp given a
  // fixed (threadIdxy, threadIdxz). Note, we reuse
  // `maybeThreadIdxxOp` here because we later want to replace this
  // op instance by 0 without relying on CSE or canonicalizations.
  Value threadIdxx = *maybeThreadIdxxOp;

  assert(workgroupSizeX % warpSize == 0);
  if (workgroupSizeX != warpSize) {
    // Add a guard for `threadIdxx < warp size` around the
    // WarpExecuteOnLane0Op.
    Value predicate = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, threadIdxx,
        arith::ConstantIndexOp::create(rewriter, loc, warpSize));
    // Note: return-less IfOp is built with a terminator, no need to
    // add one.
    auto newIfOp =
        scf::IfOp::create(rewriter, loc, predicate, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&newIfOp.getThenRegion().front());
  }
  auto warpOp = gpu::WarpExecuteOnLane0Op::create(rewriter, loc, TypeRange(),
                                                  threadIdxx, warpSize);

  // Move the code from the previous ifOp to the
  // WarpExecuteOnLane0Op.
  Block &sourceBlock = ifOp.getThenRegion().front();
  Block &targetBlock = warpOp.getWarpRegion().front();
  Block::iterator insertionPoint = targetBlock.begin();
  targetBlock.getOperations().splice(insertionPoint,
                                     sourceBlock.getOperations(),
                                     sourceBlock.without_terminator().begin(),
                                     sourceBlock.without_terminator().end());
  rewriter.setInsertionPointToEnd(&targetBlock);
  gpu::YieldOp::create(rewriter, loc);

  // Erase old op.
  rewriter.eraseOp(ifOp);

  // This simple rewrite propagates zero in lieu of laneId within the
  // warp_execute_on_lane_0 op.
  // Atm, this **must** occur before any hoisting of code.
  // TODO: Replace this by a more robust scoped SCCP that will make
  // it more robust re. hoisting.
  (void)replaceAllUsesOfLaneWithin(rewriter, warpOp);

  // Hoist the scalar code outside of the warp region.
  // Note: moving code does not require a listener.
  vector::moveScalarUniformCode(warpOp);

  return VectorDistributionResult{warpOp};
}

DiagnosedSilenceableFailure
transform_dialect::VectorToWarpExecuteOnLane0Op::applyToOne(
    transform::TransformRewriter &rewriter, scf::IfOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto funcOp = target->getParentOfType<mlir::FunctionOpInterface>();

  std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
      getWorkgroupSize(funcOp);
  // TODO: Pervasive 3 constant in IREE.
  if (!maybeWorkgroupSize || maybeWorkgroupSize->empty()) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "export op must have workgroup_size attribute set "
              "with 3 entries "
              "--- the transform is not applied";
  }

  int64_t workgroupSizeX = (*maybeWorkgroupSize)[0];
  int64_t warpSize = getWarpSize();
  if (workgroupSizeX % warpSize != 0) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "vector distribution requires workgroup size for x to "
              "be a "
           << "multiple of the warp size: " << workgroupSizeX << " vs "
           << warpSize << " --- the transform is not applied";
  }

  Location loc = target->getLoc();
  rewriter.setInsertionPoint(target);
  FailureOr<VectorDistributionResult> vectorDistributionResult =
      rewriteScfIfAsWarpExecuteOnLane0(rewriter, loc, target, workgroupSizeX,
                                       warpSize);
  if (failed(vectorDistributionResult)) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return mlir::emitSilenceableFailure(
        target, "scf::ifOp needs to be predicated on threadIdx.x == 0 "
                "--- the transform is not applied");
  }

  results.push_back(vectorDistributionResult->warpOp);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// VectorWarpDistributionOp.
//===---------------------------------------------------------------------===//
void transform_dialect::VectorWarpDistributionOp::build(OpBuilder &builder,
                                                        OperationState &result,
                                                        Value target) {
  result.addOperands(target);
}

void transform_dialect::VectorWarpDistributionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        gpu::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(),
                        MemRefLayoutAttrInterface{}, addressSpaceAttr);
  } else {
    memrefType = MemRefType::get({1}, type, MemRefLayoutAttrInterface{},
                                 addressSpaceAttr);
  }
  return memref::AllocOp::create(builder, loc, memrefType);
}

/// Return a value yielded by `warpOp` which satisfies the filter lambda
/// condition and is not dead.
static OpOperand *getWarpResult(gpu::WarpExecuteOnLane0Op warpOp,
                                function_ref<bool(Operation *)> fn) {
  auto yield = cast<gpu::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty()) {
        return &yieldOperand;
      }
    }
  }
  return {};
}

namespace {
/// Sink out load op feeding into a warp op yield.
/// ```
/// %0 = vector.warp_execute_on_lane_0(%arg0) -> (f32) {
///   ...
//    %2 = memref.load %src[%c0] : memref<1024xf32>
///   vector.yield %2 : f32
/// }
/// ```
/// To
/// ```
/// %dead = vector.warp_execute_on_lane_0(%arg0) -> (f32) {
///   ...
//    %2 = memref.load %src[%c0] : memref<1024xf32>
///   vector.yield %2 : f32
/// }
/// gpu.synchronize
/// %0 = memref.load %src[%c0] : memref<1024xf32>
struct WarpOpLoad : public OpRewritePattern<gpu::WarpExecuteOnLane0Op> {
  using Base::Base;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(warpOp, llvm::IsaPred<memref::LoadOp>);
    if (!operand) {
      return failure();
    }
    auto load = operand->get().getDefiningOp<memref::LoadOp>();
    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    auto indices = llvm::to_vector_of<Value>(load.getIndices());
    if (!indices.empty()) {
      return failure();
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    // TODO: generalize this.
    // options.warpSyncronizationFn currently must take a
    // WarpExecuteOnLane0Op which we don't have here.
    gpu::BarrierOp::create(rewriter, load.getLoc(), load.getMemref());
    Value newRead = memref::LoadOp::create(rewriter, load.getLoc(),
                                           distributedVal.getType(),
                                           load.getMemref(), indices);

    // The result type of WarpExecuteOnLane0Op may or may not match
    // the yielded type depending on whether the op has "broadcast"
    // behavior (see the doc of WarpExecuteOnLane0Op).
    for (OpOperand &use : distributedVal.getUses()) {
      rewriter.startOpModification(use.getOwner());
      Value replacement = newRead;
      if (use.get().getType() != newRead.getType()) {
        replacement = vector::BroadcastOp::create(rewriter, load.getLoc(),
                                                  use.get().getType(), newRead);
      }
      use.getOwner()->setOperand(use.getOperandNumber(), replacement);
      rewriter.finalizeOpModification(use.getOwner());
    }
    return success();
  }
};

/// Shared memory allocations are representated as AllocOp in IREE but they
/// really have the semantic of global variables. Therefore hoisting them is
/// always correct for static allocations.
struct HoistSharedMemoryAlloc : public OpRewritePattern<memref::AllocOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    if (!iree_compiler::hasSharedMemoryAddressSpace(alloc.getType())) {
      return failure();
    }
    auto warpParent = alloc->getParentOfType<gpu::WarpExecuteOnLane0Op>();
    if (!warpParent) {
      return failure();
    }
    alloc->moveBefore(warpParent);
    // Conservatively move the dealloc after the warpOp. This may
    // extend the liverange of the allocation but is always correct.
    for (Operation *user : alloc->getUsers()) {
      if (isa<memref::DeallocOp>(user)) {
        user->moveAfter(warpParent);
      }
    }
    return success();
  }
};

} // namespace

static void populateMultiReductionLoweringPatterns(Operation *target,
                                                   RewritePatternSet &patterns,
                                                   PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction, benefit);
}

static AffineMap simpleDistributionFunction(Value val) {
  AffineMap map = AffineMap::get(val.getContext());
  auto vecType = dyn_cast<VectorType>(val.getType());
  if (!vecType) {
    return map;
  }
  // Create a map (d0, d1) -> (d1) to distribute along the inner
  // dimension. Once we support n-d distribution we can add more
  // complex cases.
  int64_t vecRank = vecType.getRank();
  OpBuilder builder(val.getContext());
  map = AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));
  return map;
}

static void populateVectorTransferWriteDistribution(Operation *target,
                                                    RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populateDistributeTransferWriteOpPatterns(
      patterns, simpleDistributionFunction, /*maxNumElementsToExtract=*/1,
      benefit);
}

static Value simpleWarpShuffleFunction(Location loc, OpBuilder &builder,
                                       Value val, Value srcIdx,
                                       int64_t warpSz) {
  assert((val.getType().isF32() || val.getType().isInteger(32)) &&
         "unsupported shuffle type");
  Type i32Type = builder.getIntegerType(32);
  Value srcIdxI32 = arith::IndexCastOp::create(builder, loc, i32Type, srcIdx);
  Value warpSzI32 = arith::ConstantOp::create(
      builder, loc, builder.getIntegerAttr(i32Type, warpSz));
  Value result = gpu::ShuffleOp::create(builder, loc, val, srcIdxI32, warpSzI32,
                                        gpu::ShuffleMode::IDX)
                     .getResult(0);
  return result;
}

static void populatePropagateVectorDistribution(Operation *target,
                                                RewritePatternSet &patterns,
                                                PatternBenefit benefit,
                                                unsigned subgroupSize) {
  auto groupReductionFn =
      [subgroupSize](Location loc, OpBuilder &builder, Value input,
                     vector::CombiningKind kind, uint32_t size) {
        return mlir::iree_compiler::emitGPUGroupReduction(
            loc, builder, input, kind, size, subgroupSize,
            /*expandSubgroupReduce=*/true);
      };
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populatePropagateWarpVectorDistributionPatterns(
      patterns, simpleDistributionFunction, simpleWarpShuffleFunction, benefit);
  vector::populateDistributeReduction(patterns, groupReductionFn, benefit);
  patterns.add<WarpOpLoad, HoistSharedMemoryAlloc>(target->getContext(),
                                                   benefit);
}

static void warpSyncronizationFn(Location loc, OpBuilder &builder,
                                 gpu::WarpExecuteOnLane0Op warpOp) {
  // The memory we must synchronize on is in shared memory.
  gpu::BarrierOp::create(builder, loc, gpu::AddressSpace::Workgroup);
};

static void populateWarpExecuteOnLane0ToScf(
    Operation *target, RewritePatternSet &patterns,
    const vector::WarpExecuteOnLane0LoweringOptions &options,
    PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options,
                                                      benefit);
}

DiagnosedSilenceableFailure
transform_dialect::VectorWarpDistributionOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto subgroupSize = getSubgroupSize(target);
  if (!subgroupSize) {
    target->emitOpError(
        "could not extract subgroup size from IREE::Codegen::TranslationInfo");
    return emitDefaultDefiniteFailure(target);
  }

  // TODO: Hook up into the ApplyPatternOp in CommonExtensions.cpp to
  // automatically get listening capabilities.

  MLIRContext *ctx = target->getContext();
  // MultiReduction lowering is necessary until we have explicit
  // support for distributing that op.
  RewritePatternSet preProcessingPatterns(ctx);
  populateMultiReductionLoweringPatterns(target, preProcessingPatterns,
                                         /*benefit=*/1);
  vector::ShapeCastOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  vector::BroadcastOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  vector::ExtractOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  ErrorCheckingTrackingListener listener(state, *this);
  auto checkErrors = llvm::scope_exit([&]() {
    // The TrackingListener API makes checking for errors mandatory. It is safe
    // to drop payload ops during this transform, so we can ignore all errors.
    (void)listener.checkAndResetError();
  });
  GreedyRewriteConfig config;
  config.setListener(&listener);
  if (failed(applyPatternsGreedily(target, std::move(preProcessingPatterns),
                                   config))) {
    return mlir::emitDefiniteFailure(target,
                                     "multi-reduce patterns failed to apply");
  }

  RewritePatternSet patterns(ctx);
  populateVectorTransferWriteDistribution(target, patterns,
                                          /*benefit=*/2);
  unsigned subgroupSizeU = static_cast<unsigned>(subgroupSize.value());
  populatePropagateVectorDistribution(target, patterns,
                                      /*benefit=*/1, subgroupSizeU);
  if (failed(applyPatternsGreedily(target, std::move(patterns), config))) {
    return mlir::emitDefiniteFailure(
        target, "warp distribution patterns failed to apply");
  }

  RewritePatternSet endPatterns(ctx);
  vector::WarpExecuteOnLane0LoweringOptions options;
  options.warpAllocationFn = allocateGlobalSharedMemory;
  options.warpSyncronizationFn = warpSyncronizationFn;
  populateWarpExecuteOnLane0ToScf(target, endPatterns, options,
                                  /*benefit=*/0);
  if (failed(applyPatternsGreedily(target, std::move(endPatterns), config))) {
    return mlir::emitDefiniteFailure(
        target, "warp execute on lane 0 to scf patterns failed to apply");
  }

  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::VectorToMMAConversionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform_dialect::VectorToMMAConversionOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError(
        "applies only to isolated-from-above targets because it "
        "needs to apply "
        "patterns greedily");
    return emitDefaultDefiniteFailure(target);
  }

  auto funcOp = dyn_cast<mlir::FunctionOpInterface>(target);
  if (!funcOp) {
    target->emitOpError("Must apply to a func op");
    return emitDefaultDefiniteFailure(target);
  }

  if (!(getUseMmaSync() ^ getUseWmma())) {
    target->emitOpError(
        "Exactly one of use_mma_sync or use_wmma must be specified");
    return emitDefaultDefiniteFailure(target);
  }

  MLIRContext *ctx = target->getContext();
  ErrorCheckingTrackingListener listener(state, *this);
  GreedyRewriteConfig config;
  config.setListener(&listener);

  // Unrolling to native vector size must have previously occurred.
  // TODO: Add pattern to propagate the extract through the scf.for
  // ops. Convert slice of contract operations to mma_sync/wmma ops.
  RewritePatternSet patterns(ctx);
  mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  populatePrepareVectorToMMAPatterns(patterns, getUseMmaSync());
  if (failed(applyPatternsGreedily(target, std::move(patterns), config))) {
    target->emitOpError("vector to mma preparation patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }

  DEBUG_WITH_TYPE(DEBUG_VECTOR_TO_MMA, {
    DBGS_VECTOR_TO_MMA() << "after cast away vector leading one dim patterns:\n"
                         << *target << "\n";
  });

  auto diag = DiagnosedSilenceableFailure::success();
  if (getUseWmma()) {
    if (failed(convertVectorToMMAOps(rewriter, target))) {
      return mlir::emitDefiniteFailure(
          target, "vector to wmma patterns failed to apply");
    }
    return listener.checkAndResetError();
  }

  if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, funcOp))) {
    return mlir::emitDefiniteFailure(target,
                                     "vector to mma patterns failed to apply");
  }

  DEBUG_WITH_TYPE(DEBUG_VECTOR_TO_MMA,
                  {
                    DBGS_VECTOR_TO_MMA()
                        << "after convert vector to NVVM compatible MMA sync:\n"
                        << *target << "\n";
                  });

  // Using TF32 for Float.
  RewritePatternSet f32ToTF32patterns(funcOp.getContext());
  nvgpu::populateMmaSyncF32ToTF32Patterns(f32ToTF32patterns,
                                          nvgpu::MmaSyncF32Lowering::TF32);
  if (failed(applyPatternsGreedily(funcOp, std::move(f32ToTF32patterns),
                                   config))) {
    return mlir::emitDefiniteFailure(
        target, "vector to mma F32ToTF32 patterns failed to apply");
  }

  return listener.checkAndResetError();
}

//===----------------------------------------------------------------------===//
// PromoteOperandsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::PromoteOperandsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  Location loc = target->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(target);
  SmallVector<int64_t> indices = llvm::to_vector(getIndices());
  int64_t numOperands = target->getNumOperands();

  results.push_back(target);
  bufferization::BufferizationOptions options;
  bufferization::BufferizationState bufferizationState;
  for (int64_t index : indices) {
    if ((index >= 0) && (index < numOperands)) {
      FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
          rewriter, loc, target->getOperand(index), options,
          bufferizationState);
      if (failed(ret)) {
        return emitDefaultDefiniteFailure(target)
               << "failed to promote operand";
      }
      target->setOperand(index, ret.value());
      results.push_back(ret.value().getDefiningOp());
    } else {
      return emitDefaultDefiniteFailure(target) << "invalid index specified";
    }
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// PipelineSharedMemoryCopiesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::PipelineSharedMemoryCopiesOp::applyToOne(
    transform::TransformRewriter &rewriter, scf::ForOp forOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  int64_t depth(getDepth());
  auto schedule = getUseMmaSync()
                      ? PipeliningSchedulingStrategy::nvidiaTensorCore
                      : PipeliningSchedulingStrategy::loadGlobalStage0;
  FailureOr<scf::ForOp> pipelinedFor = iree_compiler::pipelineSharedMemoryCopy(
      rewriter, forOp, schedule, getPeelEpilogue(), depth);
  if (failed(pipelinedFor)) {
    results.push_back(forOp);
    return DiagnosedSilenceableFailure::success();
  }
  results.push_back(pipelinedFor.value());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// SynchronizeLoopOp
//===----------------------------------------------------------------------===//

void transform_dialect::SynchronizeLoopOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getForOpMutable(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform_dialect::SynchronizeLoopOp::applyToOne(
    transform::TransformRewriter &rewriter, scf::ForOp forOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPointAfter(forOp);
  gpu::BarrierOp::create(rewriter, forOp.getLoc());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// CreateAsyncGroupsOp.
//===---------------------------------------------------------------------===//

void transform_dialect::CreateAsyncGroupsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform_dialect::CreateAsyncGroupsOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  iree_compiler::createAsyncGroups(
      rewriter, cast<mlir::FunctionOpInterface>(target), getUseMmaSync());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ReorderTransposeOp
//===---------------------------------------------------------------------===//
DiagnosedSilenceableFailure transform_dialect::ReorderTransposeOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  iree_compiler::reorderTranspose(rewriter,
                                  cast<mlir::FunctionOpInterface>(target));
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::EliminateGpuBarriersOp::build(OpBuilder &builder,
                                                      OperationState &state,
                                                      Value target) {
  build(builder, state, target.getType(), target);
}

DiagnosedSilenceableFailure
transform_dialect::EliminateGpuBarriersOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  RewritePatternSet patterns(target.getContext());
  // This used to be in this file, now it's upstream.
  mlir::populateGpuEliminateBarriersPatterns(patterns);
  ErrorCheckingTrackingListener listener(state, *this);
  auto checkErrors = llvm::scope_exit([&]() {
    // The TrackingListener API makes checking for errors mandatory. It is safe
    // to drop payload ops during this transform, so we can ignore all errors.
    (void)listener.checkAndResetError();
  });
  GreedyRewriteConfig config;
  config.setListener(&listener);
  if (failed(applyPatternsGreedily(target, std::move(patterns), config))) {
    return emitDefaultSilenceableFailure(target);
  }

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform_dialect::PackSharedMemoryAllocOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  packSharedMemoryAlloc(target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::PackSharedMemoryAllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// PrefetchSharedMemoryCopiesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::PrefetchSharedMemoryCopiesOp::applyToOne(
    transform::TransformRewriter &rewriter, scf::ForOp forOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  FailureOr<scf::ForOp> pipelinedFor =
      iree_compiler::prefetchSharedMemoryCopy(rewriter, forOp);

  if (failed(pipelinedFor)) {
    results.push_back(forOp);
    return DiagnosedSilenceableFailure::success();
  }
  results.push_back(pipelinedFor.value());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform_dialect::AMDGPUDistributeVectorsOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  TransformVectorLayoutOptions options(target, !getTestConversion());
  RewritePatternSet patterns(target.getContext());

  rewriter.setInsertionPointToStart(&target.getFunctionBody().front());
  Value laneId =
      gpu::ThreadIdOp::create(rewriter, target.getLoc(), gpu::Dimension::x);
  int64_t subgroupSize = getSubgroupSize();
  ArrayRef<int64_t> workgroupSize = getWorkgroupSize();

  populateGPUDistributionPatterns(patterns);
  populateGPUDistributeNestedLayoutAttrPatterns(patterns, laneId, subgroupSize,
                                                workgroupSize);
  if (failed(distributeVectorOps(target, patterns, options))) {
    return emitDefaultSilenceableFailure(target);
  }
  // TODO: The consumption of the target handle is only required because the
  // transform dialect interpreter will crash without it. This op should not
  // need to invalidate the handle.
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::AMDGPUDistributeVectorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// CreateMatmulMfmaTileSizesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::CreateMatmulMfmaTileSizesOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto payload = state.getPayloadOps(getTarget());
  Operation *target = *payload.begin();
  ArrayRef<int64_t> shape =
      cast<ShapedType>(target->getResult(0).getType()).getShape();
  if (shape.size() != 2) {
    return emitDefiniteFailure() << "matmul shape size should be 2";
  }
  SmallVector<int64_t> sz0, sz1;
  SmallVector<Attribute> paramsArray0, paramsArray1;

  // TODO: cover all shape cases
  // (sz0[0] / sz1[0]) * (sz0[1] / sz1[1]) = 8
  if (shape == ArrayRef<int64_t>({8192, 320})) {
    sz0 = {512, 64};
    sz1 = {64, 64};
  } else if (shape[0] == 128) {
    if (shape[1] % 64 != 0) {
      return emitDefiniteFailure() << "dim #1 should be divisible by 64";
    }
    sz0 = {128, 64};
    sz1 = {32, 32};
  } else {
    // Default tiling configuration.
    if (shape[0] % 256 != 0 || shape[1] % 128 != 0) {
      return emitDefiniteFailure() << "dim #0 should be divisible by 256 and "
                                      "dim #1 should be divisible by 128";
    }
    sz0 = {256, 128};
    sz1 = {64, 64};
  }
  for (auto i : sz0) {
    paramsArray0.push_back(rewriter.getIntegerAttr(rewriter.getIndexType(), i));
  }
  for (auto i : sz1) {
    paramsArray1.push_back(rewriter.getIntegerAttr(rewriter.getIndexType(), i));
  }

  results.setParams(cast<OpResult>(getResult(0)), paramsArray0);
  results.setParams(cast<OpResult>(getResult(1)), paramsArray0);
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.cpp.inc"
