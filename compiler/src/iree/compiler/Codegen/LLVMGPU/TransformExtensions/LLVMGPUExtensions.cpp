// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LLVMGPUExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
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
#define LDBG(X) LLVM_DEBUG(dbgs() << '[' << DEBUG_TYPE << "] " << X)
#define DBGS_ALIAS() (dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")
#define DBGS_VECTOR_TO_MMA() (dbgs() << '[' << DEBUG_VECTOR_TO_MMA << "] ")

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

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
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  FailureOr<IREE::HAL::ExecutableExportOp> maybeExportOp =
      getEntryPoint(target);
  if (failed(maybeExportOp)) {
    state.getTopLevel()->emitOpError("no IREE::HAL::ExecutableExportOp found");
    return emitDefaultDefiniteFailure(target);
  }
  IREE::HAL::ExecutableExportOp exportOp = *maybeExportOp;

  auto transformOp = cast<transform::TransformOpInterface>(getOperation());

  rewriter.setInsertionPointToStart(&target.getBody().front());
  DiagnosedSilenceableFailure diag =
      mlir::transform::gpu::mapNestedForallToThreadsImpl(
          rewriter, transformOp, target, getWorkgroupDims(), getSubgroupSize(),
          true);
  if (!diag.succeeded())
    return diag;
  auto newAttr = rewriter.getIndexArrayAttr(getWorkgroupDims());
  auto subgroupSizeAttr = rewriter.getIndexAttr(getSubgroupSize());
  rewriter.startOpModification(exportOp);
  exportOp->setAttr(exportOp.getWorkgroupSizeAttrName(), newAttr);
  exportOp->setAttr(exportOp.getSubgroupSizeAttrName(), subgroupSizeAttr);
  rewriter.finalizeOpModification(exportOp);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::MapNestedForallToGpuThreadsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
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
                           vector::WarpExecuteOnLane0Op executeOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(executeOp);
  Value zero = b.create<arith::ConstantIndexOp>(executeOp.getLoc(), 0);
  b.setInsertionPointToStart(&executeOp.getWarpRegion().front());
  Value laneId = executeOp.getLaneid();
  bool applied = false;
  for (Operation *user : llvm::make_early_inc_range(laneId.getUsers())) {
    if (!executeOp->isProperAncestor(user))
      continue;
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
      !ifOp.getElseRegion().empty())
    return failure();
  auto pred = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!pred)
    return failure();
  auto EQ = arith::CmpIPredicate::eq;
  auto SLT = arith::CmpIPredicate::slt;
  auto SLE = arith::CmpIPredicate::sle;
  auto ULT = arith::CmpIPredicate::ult;
  auto ULE = arith::CmpIPredicate::ule;
  if (auto threadIdOp = pred.getLhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.getDimension() != gpu::Dimension::x)
      return failure();
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SLE && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == ULE && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SLT && isConstantIntValue(pred.getRhs(), 1))
      return threadIdOp;
    if (pred.getPredicate() == ULT && isConstantIntValue(pred.getRhs(), 1))
      return threadIdOp;
  }
  auto SGT = arith::CmpIPredicate::sgt;
  auto SGE = arith::CmpIPredicate::sge;
  auto UGT = arith::CmpIPredicate::ugt;
  auto UGE = arith::CmpIPredicate::uge;
  if (auto threadIdOp = pred.getRhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.getDimension() != gpu::Dimension::x)
      return failure();
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SGE && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == UGE && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SGT && isConstantIntValue(pred.getLhs(), 1))
      return threadIdOp;
    if (pred.getPredicate() == UGT && isConstantIntValue(pred.getLhs(), 1))
      return threadIdOp;
  }
  return failure();
}

struct VectorDistributionResult {
  vector::WarpExecuteOnLane0Op warpOp;
};

static FailureOr<VectorDistributionResult>
rewriteScfIfAsWarpExecuteOnLane0(RewriterBase &rewriter, Location loc,
                                 scf::IfOp ifOp, int64_t workgroupSizeX,
                                 int64_t warpSize) {
  // Bail if cond is not `if (threadIdx.x == 0)`.
  FailureOr<gpu::ThreadIdOp> maybeThreadIdxxOp =
      isThreadIdxxZeroPredicate(ifOp);
  if (failed(maybeThreadIdxxOp))
    return failure();

  // All the code below will be executed on a single warp given a
  // fixed (threadIdxy, threadIdxz). Note, we reuse
  // `maybeThreadIdxxOp` here because we later want to replace this
  // op instance by 0 without relying on CSE or canonicalizations.
  Value threadIdxx = *maybeThreadIdxxOp;

  assert(workgroupSizeX % warpSize == 0);
  if (workgroupSizeX != warpSize) {
    // Add a guard for `threadIdxx < warp size` around the
    // WarpExecuteOnLane0Op.
    Value predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadIdxx,
        rewriter.create<arith::ConstantIndexOp>(loc, warpSize));
    // Note: return-less IfOp is built with a terminator, no need to
    // add one.
    auto newIfOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&newIfOp.getThenRegion().front());
  }
  auto warpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
      loc, TypeRange(), threadIdxx, warpSize);

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
  rewriter.create<vector::YieldOp>(loc);

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

// TODO: Refactor in a generic util that can be reused.
static HAL::ExecutableExportOp
getExecutableExportOpForFunc(HAL::ExecutableVariantOp halExecutableVariantOp,
                             func::FuncOp funcOp) {
  if (!halExecutableVariantOp || !funcOp)
    return {};
  HAL::ExecutableExportOp exportOp;
  halExecutableVariantOp->walk([&](HAL::ExecutableExportOp op) {
    if (op.getSymName() != funcOp.getName())
      return WalkResult::advance();
    exportOp = op;
    return WalkResult::interrupt();
  });
  return exportOp;
}

DiagnosedSilenceableFailure
transform_dialect::VectorToWarpExecuteOnLane0Op::applyToOne(
    transform::TransformRewriter &rewriter, scf::IfOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!isa<HAL::ExecutableOp, HAL::ExecutableVariantOp>(state.getTopLevel())) {
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(state.getTopLevel())
           << "requires HAL::ExecutableOp or "
              "HAL::ExecutableVariantOp toplevel "
              "so that IR is properly isolated. This is required so "
              "we can "
              "safely inspect the HAL::ExecutableExportOp under "
              "multi-threaded "
              "pass assumptions.";
  }

  auto halExecutableVariantOp =
      target->getParentOfType<HAL::ExecutableVariantOp>();
  auto funcOp = target->getParentOfType<func::FuncOp>();
  HAL::ExecutableExportOp exportOp =
      getExecutableExportOpForFunc(halExecutableVariantOp, funcOp);
  if (!halExecutableVariantOp || !funcOp || !exportOp) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "export op is missing --- the transform is not "
              "applied";
  }

  std::optional<ArrayAttr> maybeAttr = exportOp.getWorkgroupSize();
  // TODO: Pervasive 3 constant in IREE.
  if (!maybeAttr || maybeAttr->size() != 3) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "export op must have workgroup_size attribute set "
              "with 3 entries "
              "--- the transform is not applied";
  }

  int64_t workgroupSizeX = llvm::cast<IntegerAttr>((*maybeAttr)[0]).getInt();
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
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(),
                        MemRefLayoutAttrInterface{}, addressSpaceAttr);
  } else {
    memrefType = MemRefType::get({1}, type, MemRefLayoutAttrInterface{},
                                 addressSpaceAttr);
  }
  return builder.create<memref::AllocOp>(loc, memrefType);
}

/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
static OpOperand *getWarpResult(vector::WarpExecuteOnLane0Op warpOp,
                                function_ref<bool(Operation *)> fn) {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
}

namespace {
/// Pattern to convert InsertElement to broadcast, this is a workaround
/// until MultiDimReduction distribution is supported.
class InsertElementToBroadcast final
    : public OpRewritePattern<vector::InsertElementOp> {
public:
  using OpRewritePattern<vector::InsertElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertElementOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp.getDestVectorType().getNumElements() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getSource());
    return success();
  }
};

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
struct WarpOpLoad : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<memref::LoadOp>(op); });
    if (!operand)
      return failure();
    auto load = operand->get().getDefiningOp<memref::LoadOp>();
    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    SmallVector<Value> indices(load.getIndices().begin(),
                               load.getIndices().end());
    if (!indices.empty())
      return failure();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    // TODO: generalize this.
    // options.warpSyncronizationFn currently must take a
    // WarpExecuteOnLane0Op which we don't have here.
    rewriter.create<gpu::BarrierOp>(load.getLoc());
    Value newRead = rewriter.create<memref::LoadOp>(
        load.getLoc(), distributedVal.getType(), load.getMemref(), indices);

    // The result type of WarpExecuteOnLane0Op may or may not match
    // the yielded type depending on whether the op has "broadcast"
    // behavior (see the doc of WarpExecuteOnLane0Op).
    for (OpOperand &use : distributedVal.getUses()) {
      rewriter.startOpModification(use.getOwner());
      Value replacement = newRead;
      if (use.get().getType() != newRead.getType()) {
        replacement = rewriter.create<vector::BroadcastOp>(
            load.getLoc(), use.get().getType(), newRead);
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
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    if (!iree_compiler::hasSharedMemoryAddressSpace(alloc.getType()))
      return failure();
    auto warpParent = alloc->getParentOfType<vector::WarpExecuteOnLane0Op>();
    if (!warpParent)
      return failure();
    alloc->moveBefore(warpParent);
    // Conservatively move the dealloc after the warpOp. This may
    // extend the liverange of the allocation but is always correct.
    for (Operation *user : alloc->getUsers()) {
      if (isa<memref::DeallocOp>(user))
        user->moveAfter(warpParent);
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
  patterns.add<InsertElementToBroadcast>(target->getContext(), benefit);
}

static AffineMap simpleDistributionFunction(Value val) {
  AffineMap map = AffineMap::get(val.getContext());
  auto vecType = llvm::dyn_cast<VectorType>(val.getType());
  if (!vecType)
    return map;
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
  Value srcIdxI32 = builder.create<arith::IndexCastOp>(loc, i32Type, srcIdx);
  Value warpSzI32 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(i32Type, warpSz));
  Value result = builder
                     .create<gpu::ShuffleOp>(loc, val, srcIdxI32, warpSzI32,
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
                                 vector::WarpExecuteOnLane0Op warpOp) {
  builder.create<gpu::BarrierOp>(loc);
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
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  FailureOr<IREE::HAL::ExecutableExportOp> maybeExportOp =
      getEntryPoint(target);
  if (failed(maybeExportOp)) {
    state.getTopLevel()->emitOpError("no IREE::HAL::ExecutableExportOp found");
    return emitDefaultDefiniteFailure(target);
  }
  IREE::HAL::ExecutableExportOp exportOp = *maybeExportOp;

  std::optional<llvm::APInt> subgroupSize = exportOp.getSubgroupSize();
  if (!subgroupSize) {
    state.getTopLevel()->emitOpError(
        "could not extract subgroup size from IREE::HAL::ExecutableExportOp");
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
  auto checkErrors = llvm::make_scope_exit([&]() {
    // The TrackingListener API makes checking for errors mandatory. It is safe
    // to drop payload ops during this transform, so we can ignore all errors.
    (void)listener.checkAndResetError();
  });
  GreedyRewriteConfig config;
  config.listener = &listener;
  if (failed(applyPatternsAndFoldGreedily(
          target, std::move(preProcessingPatterns), config))) {
    return mlir::emitDefiniteFailure(target,
                                     "multi-reduce patterns failed to apply");
  }

  RewritePatternSet patterns(ctx);
  populateVectorTransferWriteDistribution(target, patterns,
                                          /*benefit=*/2);
  populatePropagateVectorDistribution(target, patterns,
                                      /*benefit=*/1,
                                      subgroupSize->getSExtValue());
  if (failed(
          applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
    return mlir::emitDefiniteFailure(
        target, "warp distribution patterns failed to apply");
  }

  RewritePatternSet endPatterns(ctx);
  vector::WarpExecuteOnLane0LoweringOptions options;
  options.warpAllocationFn = allocateGlobalSharedMemory;
  options.warpSyncronizationFn = warpSyncronizationFn;
  populateWarpExecuteOnLane0ToScf(target, endPatterns, options,
                                  /*benefit=*/0);
  if (failed(applyPatternsAndFoldGreedily(target, std::move(endPatterns),
                                          config))) {
    return mlir::emitDefiniteFailure(
        target, "warp execute on lane 0 to scf patterns failed to apply");
  }

  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::VectorToMMAConversionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
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

  auto funcOp = dyn_cast<func::FuncOp>(target);
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
  config.listener = &listener;

  // Unrolling to native vector size must have previously occurred.
  // TODO: Add pattern to propagate the extract through the scf.for
  // ops. Convert slice of contract operations to mma_sync/wmma ops.
  RewritePatternSet patterns(ctx);
  mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  populatePrepareVectorToMMAPatterns(patterns, getUseMmaSync());
  if (failed(
          applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
    target->emitOpError("vector to mma preparation patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }

  DEBUG_WITH_TYPE(DEBUG_VECTOR_TO_MMA, {
    DBGS_VECTOR_TO_MMA() << "after cast away vector leading one dim patterns:\n"
                         << *target << "\n";
  });

  auto diag = DiagnosedSilenceableFailure::success();
  if (getUseWmma()) {
    if (failed(convertVectorToMMAOps(rewriter, target)))
      return mlir::emitDefiniteFailure(
          target, "vector to wmma patterns failed to apply");
    return listener.checkAndResetError();
  }

  if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, funcOp)))
    return mlir::emitDefiniteFailure(target,
                                     "vector to mma patterns failed to apply");

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
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(f32ToTF32patterns),
                                          config)))
    return mlir::emitDefiniteFailure(
        target, "vector to mma F32ToTF32 patterns failed to apply");

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
  for (int64_t index : indices) {
    if ((index >= 0) && (index < numOperands)) {
      FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
          rewriter, loc, target->getOperand(index), options);
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
  transform::onlyReadsHandle(getForOp(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform_dialect::SynchronizeLoopOp::applyToOne(
    transform::TransformRewriter &rewriter, scf::ForOp forOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPointAfter(forOp);
  rewriter.create<gpu::BarrierOp>(forOp.getLoc());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// CreateAsyncGroupsOp.
//===---------------------------------------------------------------------===//

void transform_dialect::CreateAsyncGroupsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform_dialect::CreateAsyncGroupsOp::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  iree_compiler::createAsyncGroups(rewriter, cast<func::FuncOp>(target),
                                   getUseMmaSync());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// LayoutAnalysisAndDistributionOp.
//===---------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform_dialect::LayoutAnalysisAndDistributionOp::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  iree_compiler::doLayoutAnalysisAndDistribution(rewriter,
                                                 cast<func::FuncOp>(target));
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ReorderTransposeOp
//===---------------------------------------------------------------------===//
DiagnosedSilenceableFailure transform_dialect::ReorderTransposeOp::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  iree_compiler::reorderTranspose(rewriter, cast<func::FuncOp>(target));
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// EliminateGpuBarriersOp
//===---------------------------------------------------------------------===//

/// Returns `true` if the op is known not to have any side effects, but does not
/// implement the MemoryEffectsOpInterface in the suitable way.
static bool isKnownNoEffectsOpWithoutInterface(Operation *op) {
  // memref::AssumeAlignment is conceptually pure, but marking it as such would
  // make DCE immediately remove it.
  return isa<memref::AssumeAlignmentOp>(op);
}

/// Returns `true` if the op is defines the parallel region that is subject to
/// barrier synchronization.
static bool isParallelRegionBoundary(Operation *op) {
  if (op->hasAttr("__parallel_region_boundary_for_test"))
    return true;

  // We consider functions inside executable variants that have the same symbol
  // name as an export symbol.
  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func)
    return false;
  auto parent = op->getParentOfType<ModuleOp>();
  if (!parent)
    return false;
  auto variant = parent->getParentOfType<HAL::ExecutableVariantOp>();
  if (!variant)
    return false;
  WalkResult result = variant.walk([&](HAL::ExecutableExportOp exportOp) {
    if (exportOp.getSymNameAttr() == func.getNameAttr())
      return WalkResult::interrupt();
    return WalkResult::skip();
  });
  return result.wasInterrupted();
}

/// Returns `true` if the op behaves like a sequential loop, e.g., the control
/// flow "wraps around" from the end of the body region back to its start.
static bool isSequentialLoopLike(Operation *op) { return isa<scf::ForOp>(op); }

/// Returns `true` if the regions of the op are guaranteed to be executed at
/// most once. Thus, if an operation in one of the nested regions of `op` is
/// executed than so are all the other operations in this region.
static bool hasSingleExecutionBody(Operation *op) {
  return isa<scf::IfOp, memref::AllocaScopeOp>(op);
}

/// Returns `true` if the operation is known to produce a pointer-like object
/// distinct from any other object produced by a similar operation. For example,
/// an allocation produces such an object.
static bool producesDistinctBase(Operation *op) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(op);
}

/// Populates `effects` with all memory effects without associating them to a
/// specific value.
static void addAllValuelessEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
}

/// Collect the memory effects of the given op in 'effects'. Returns 'true' if
/// it could extract the effect information from the op, otherwise returns
/// 'false' and conservatively populates the list with all possible effects
/// associated with no particular value or symbol.
static bool
collectEffects(Operation *op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
               bool ignoreBarriers = true) {
  // Skip over barriers to avoid infinite recursion (those barriers would ask
  // this barrier again).
  if (ignoreBarriers && isa<gpu::BarrierOp>(op))
    return true;

  // Skip over ops that we know have no effects.
  if (isKnownNoEffectsOpWithoutInterface(op))
    return true;

  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects, ignoreBarriers))
            return false;
      }
    }
    return true;
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  addAllValuelessEffects(effects);
  return false;
}

/// Collects memory effects from operations that may be executed before `op` in
/// a trivial structured control flow, e.g., without branches. Stops at the
/// parallel region boundary or at the barrier operation if `stopAtBarrier` is
/// set. Returns `true` if the memory effects added to `effects` are exact,
/// `false` if they are a conservative over-approximation. The latter means that
/// `effects` contain instances not associated with a specific value.
static bool
getEffectsBefore(Operation *op,
                 SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                 bool stopAtBarrier) {
  if (!op->getBlock())
    return true;

  // If there is a non-structured control flow, bail.
  Region *region = op->getBlock()->getParent();
  if (region && !llvm::hasSingleElement(region->getBlocks())) {
    addAllValuelessEffects(effects);
    return false;
  }

  // Collect all effects before the op.
  if (op != &op->getBlock()->front()) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (isa<gpu::BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        else
          continue;
      }
      if (!collectEffects(it, effects))
        return false;
    }
  }

  // Stop if reached the parallel region boundary.
  if (isParallelRegionBoundary(op->getParentOp()))
    return true;

  // Otherwise, keep collecting above the parent operation.
  if (!getEffectsBefore(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the op is loop-like, collect effects from the trailing operations until
  // we hit a barrier because they can executed before the current operation by
  // the previous iteration of this loop. For example, in the following loop
  //
  //   for i = ... {
  //     op1
  //     ...
  //     barrier
  //     op2
  //   }
  //
  // the operation `op2` at iteration `i` is known to be executed before the
  // operation `op1` at iteration `i+1` and the side effects must be ordered
  // appropriately.
  if (isSequentialLoopLike(op->getParentOp())) {
    // Assuming loop terminators have no side effects.
    return getEffectsBefore(op->getBlock()->getTerminator(), effects,
                            /*stopAtBarrier=*/true);
  }

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  bool conservative = false;
  if (!hasSingleExecutionBody(op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

/// Collects memory effects from operations that may be executed after `op` in
/// a trivial structured control flow, e.g., without branches. Stops at the
/// parallel region boundary or at the barrier operation if `stopAtBarrier` is
/// set. Returns `true` if the memory effects added to `effects` are exact,
/// `false` if they are a conservative over-approximation. The latter means that
/// `effects` contain instances not associated with a specific value.
static bool
getEffectsAfter(Operation *op,
                SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                bool stopAtBarrier) {
  if (!op->getBlock())
    return true;

  // If there is a non-structured control flow, bail.
  Region *region = op->getBlock()->getParent();
  if (region && !llvm::hasSingleElement(region->getBlocks())) {
    addAllValuelessEffects(effects);
    return false;
  }

  // Collect all effects after the op.
  if (op != &op->getBlock()->back())
    for (Operation *it = op->getNextNode(); it != nullptr;
         it = it->getNextNode()) {
      if (isa<gpu::BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        continue;
      }
      if (!collectEffects(it, effects))
        return false;
    }

  // Stop if reached the parallel region boundary.
  if (isParallelRegionBoundary(op->getParentOp()))
    return true;

  // Otherwise, keep collecting below the parent operation.
  if (!getEffectsAfter(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the op is loop-like, collect effects from the leading operations until
  // we hit a barrier because they can executed after the current operation by
  // the next iteration of this loop. For example, in the following loop
  //
  //   for i = ... {
  //     op1
  //     ...
  //     barrier
  //     op2
  //   }
  //
  // the operation `op1` at iteration `i` is known to be executed after the
  // operation `op2` at iteration `i-1` and the side effects must be ordered
  // appropriately.
  if (isSequentialLoopLike(op->getParentOp())) {
    if (isa<gpu::BarrierOp>(op->getBlock()->front()))
      return true;

    bool exact = collectEffects(&op->getBlock()->front(), effects);
    return getEffectsAfter(&op->getBlock()->front(), effects,
                           /*stopAtBarrier=*/true) &&
           exact;
  }

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  bool conservative = false;
  if (!hasSingleExecutionBody(op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

/// Looks through known "view-like" ops to find the base memref.
static Value getBase(Value v) {
  while (true) {
    Operation *definingOp = v.getDefiningOp();
    if (!definingOp)
      break;

    bool shouldContinue =
        TypeSwitch<Operation *, bool>(v.getDefiningOp())
            .Case<memref::CastOp, memref::SubViewOp, memref::ViewOp>(
                [&](auto op) {
                  v = op.getSource();
                  return true;
                })
            .Case<memref::TransposeOp>([&](auto op) {
              v = op.getIn();
              return true;
            })
            .Case<memref::CollapseShapeOp, memref::ExpandShapeOp>([&](auto op) {
              v = op.getSrc();
              return true;
            })
            .Default([](Operation *) { return false; });
    if (!shouldContinue)
      break;
  }
  return v;
}

/// Returns `true` if the value is defined as a function argument.
static bool isFunctionArgument(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  return arg && isa<FunctionOpInterface>(arg.getOwner()->getParentOp());
}

/// Returns the operand that the operation "propagates" through it for capture
/// purposes. That is, if the value produced by this operation is captured, then
/// so is the returned value.
static Value propagatesCapture(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case(
          [](ViewLikeOpInterface viewLike) { return viewLike.getViewSource(); })
      .Case([](CastOpInterface castLike) { return castLike->getOperand(0); })
      .Case([](memref::TransposeOp transpose) { return transpose.getIn(); })
      .Case<memref::ExpandShapeOp, memref::CollapseShapeOp>(
          [](auto op) { return op.getSrc(); })
      .Default([](Operation *) { return Value(); });
}

/// Returns `true` if the given operation is known to capture the given value,
/// `false` if it is known not to capture the given value, `nullopt` if neither
/// is known.
static std::optional<bool> getKnownCapturingStatus(Operation *op, Value v) {
  return llvm::TypeSwitch<Operation *, std::optional<bool>>(op)
      // Store-like operations don't capture the destination, but do capture
      // the value.
      .Case<memref::StoreOp, vector::TransferWriteOp>(
          [&](auto op) { return op.getValue() == v; })
      .Case<vector::StoreOp, vector::MaskedStoreOp>(
          [&](auto op) { return op.getValueToStore() == v; })
      // These operations are known not to capture.
      .Case([](memref::DeallocOp) { return false; })
      // By default, we don't know anything.
      .Default([](Operation *) { return std::nullopt; });
}

/// Returns `true` if the value may be captured by any of its users, i.e., if
/// the user may be storing this value into memory. This makes aliasing analysis
/// more conservative as it cannot assume the pointer-like value is only passed
/// around through SSA use-def.
static bool maybeCaptured(Value v) {
  SmallVector<Value> todo = {v};
  while (!todo.empty()) {
    Value v = todo.pop_back_val();
    for (Operation *user : v.getUsers()) {
      // A user that is known to only read cannot capture.
      auto iface = dyn_cast<MemoryEffectOpInterface>(user);
      if (iface) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        iface.getEffects(effects);
        if (llvm::all_of(effects,
                         [](const MemoryEffects::EffectInstance &effect) {
                           return isa<MemoryEffects::Read>(effect.getEffect());
                         })) {
          continue;
        }
      }

      // When an operation is known to create an alias, consider if the
      // source is captured as well.
      if (Value v = propagatesCapture(user)) {
        todo.push_back(v);
        continue;
      }

      std::optional<bool> knownCaptureStatus = getKnownCapturingStatus(user, v);
      if (!knownCaptureStatus || *knownCaptureStatus)
        return true;
    }
  }

  return false;
}

/// Returns true if two values may be referencing aliasing memory. This is a
/// rather naive and conservative analysis. Values defined by different
/// allocation-like operations as well as values derived from those by casts and
/// views cannot alias each other. Similarly, values defined by allocations
/// inside a function cannot alias function arguments. Global values cannot
/// alias each other or local allocations. Values that are captured, i.e.
/// themselves potentially stored in memory, are considered as aliasing with
/// everything. This seems sufficient to achieve barrier removal in structured
/// control flow, more complex cases would require a proper dataflow analysis.
static bool mayAlias(Value first, Value second) {
  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, {
    DBGS_ALIAS() << "checking aliasing between ";
    DBGS_ALIAS() << first << "\n";
    DBGS_ALIAS() << "                      and ";
    DBGS_ALIAS() << second << "\n";
  });

  first = getBase(first);
  second = getBase(second);

  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, {
    DBGS_ALIAS() << "base ";
    DBGS_ALIAS() << first << "\n";
    DBGS_ALIAS() << " and ";
    DBGS_ALIAS() << second << "\n";
  });

  // Values derived from the same base memref do alias (unless we do a more
  // advanced analysis to prove non-overlapping accesses).
  if (first == second) {
    DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, DBGS_ALIAS() << "-> do alias!\n");
    return true;
  }

  // Different globals cannot alias.
  if (auto globFirst = first.getDefiningOp<memref::GetGlobalOp>()) {
    if (auto globSecond = second.getDefiningOp<memref::GetGlobalOp>()) {
      return globFirst.getNameAttr() == globSecond.getNameAttr();
    }
  }
  if (auto subSpanFirst =
          first.getDefiningOp<HAL::InterfaceBindingSubspanOp>()) {
    if (auto subSpanSecond =
            second.getDefiningOp<HAL::InterfaceBindingSubspanOp>()) {
      return subSpanFirst.getBindingAttr() == subSpanSecond.getBindingAttr();
    }
  }

  bool isDistinct[] = {producesDistinctBase(first.getDefiningOp()),
                       producesDistinctBase(second.getDefiningOp())};
  bool isGlobal[] = {first.getDefiningOp<memref::GetGlobalOp>() != nullptr,
                     second.getDefiningOp<memref::GetGlobalOp>() != nullptr};

  // Non-equivalent distinct bases and globals cannot alias. At this point, we
  // have already filtered out based on values being equal and global name being
  // equal.
  if ((isDistinct[0] || isGlobal[0]) && (isDistinct[1] || isGlobal[1]))
    return false;

  bool isArg[] = {isFunctionArgument(first), isFunctionArgument(second)};

  // Distinct bases (allocations) cannot have been passed as an argument.
  if ((isDistinct[0] && isArg[1]) || (isDistinct[1] && isArg[0]))
    return false;

  // Non-captured base distinct values cannot conflict with another base value.
  if (isDistinct[0] && !maybeCaptured(first))
    return false;
  if (isDistinct[1] && !maybeCaptured(second))
    return false;

  // Otherwise, conservatively assume aliasing.
  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, DBGS_ALIAS() << "-> may alias!\n");
  return true;
}

/// Returns `true` if the effect may be affecting memory aliasing the value. If
/// the effect is not associated with any value, it is assumed to affect all
/// memory and therefore aliases with everything.
static bool mayAlias(MemoryEffects::EffectInstance a, Value v2) {
  if (Value v = a.getValue()) {
    return mayAlias(v, v2);
  }
  return true;
}

/// Returns `true` if the two effects may be affecting aliasing memory. If
/// an effect is not associated with any value, it is assumed to affect all
/// memory and therefore aliases with everything. Effects on different resources
/// cannot alias.
static bool mayAlias(MemoryEffects::EffectInstance a,
                     MemoryEffects::EffectInstance b) {
  if (a.getResource()->getResourceID() != b.getResource()->getResourceID())
    return false;
  if (Value v2 = b.getValue()) {
    return mayAlias(a, v2);
  } else if (Value v = a.getValue()) {
    return mayAlias(b, v);
  }
  return true;
}

/// Returns `true` if any of the "before" effect instances has a conflict with
/// any "after" instance for the purpose of barrier elimination. The effects are
/// supposed to be limited to a barrier synchronization scope. A conflict exists
/// if effects instances affect aliasing memory locations and at least on of
/// then as a write. As an exception, if the non-write effect is an allocation
/// effect, there is no conflict since we are only expected to see the
/// allocation happening in the same thread and it cannot be accessed from
/// another thread without capture (which we do handle in alias analysis).
static bool
haveConflictingEffects(ArrayRef<MemoryEffects::EffectInstance> beforeEffects,
                       ArrayRef<MemoryEffects::EffectInstance> afterEffects) {
  for (const MemoryEffects::EffectInstance &before : beforeEffects) {
    for (const MemoryEffects::EffectInstance &after : afterEffects) {
      // If cannot alias, definitely no conflict.
      if (!mayAlias(before, after))
        continue;

      // Read/read is not a conflict.
      if (isa<MemoryEffects::Read>(before.getEffect()) &&
          isa<MemoryEffects::Read>(after.getEffect())) {
        continue;
      }

      // Allocate/* is not a conflict since the allocation happens within the
      // thread context.
      // TODO: This is not the case for */Free unless the allocation happened in
      // the thread context, which we could also check for.
      if (isa<MemoryEffects::Allocate>(before.getEffect()) ||
          isa<MemoryEffects::Allocate>(after.getEffect())) {
        continue;
      }

      // In the particular case that the before effect is a free, we only have 2
      // possibilities:
      //   1. either the program is well-formed and there must be an interleaved
      //      alloc that must limit the scope of effect lookback and we can
      //      safely ignore the free -> read / free -> write and free -> free
      //      conflicts.
      //   2. either the program is ill-formed and we are in undefined behavior
      //      territory.
      if (isa<MemoryEffects::Free>(before.getEffect()))
        continue;

      // Other kinds of effects create a conflict, e.g. read-after-write.
      LLVM_DEBUG(
          DBGS() << "found a conflict between (before): " << before.getValue()
                 << " read:" << isa<MemoryEffects::Read>(before.getEffect())
                 << " write:" << isa<MemoryEffects::Write>(before.getEffect())
                 << " alloc:"
                 << isa<MemoryEffects::Allocate>(before.getEffect()) << " free:"
                 << isa<MemoryEffects::Free>(before.getEffect()) << "\n");
      LLVM_DEBUG(
          DBGS() << "and (after):                " << after.getValue()
                 << " read:" << isa<MemoryEffects::Read>(after.getEffect())
                 << " write:" << isa<MemoryEffects::Write>(after.getEffect())
                 << " alloc:" << isa<MemoryEffects::Allocate>(after.getEffect())
                 << " free:" << isa<MemoryEffects::Free>(after.getEffect())
                 << "\n");
      return true;
    }
  }

  return false;
}

namespace {
/// Barrier elimination pattern. If a barrier does not enforce any conflicting
/// pair of memory effects, including a pair that is enforced by another
/// barrier, it is unnecessary and can be removed. Adapted from
/// "High-Performance GPU-to-CPU Transpilation and Optimization via High-Level
/// Parallel Constructs" by Moses et.al. in PPoPP 2023 and implementation in
/// Polygeist.
class BarrierElimination final : public OpRewritePattern<gpu::BarrierOp> {
public:
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp barrier,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(DBGS() << "checking the necessity of: " << barrier << " "
                      << barrier.getLoc() << "\n");

    SmallVector<MemoryEffects::EffectInstance> beforeEffects;
    getEffectsBefore(barrier, beforeEffects, /*stopAtBarrier=*/true);

    SmallVector<MemoryEffects::EffectInstance> afterEffects;
    getEffectsAfter(barrier, afterEffects, /*stopAtBarrier=*/true);

    if (!haveConflictingEffects(beforeEffects, afterEffects)) {
      LLVM_DEBUG(DBGS() << "the surrounding barriers are sufficient, removing "
                        << barrier << "\n");
      rewriter.eraseOp(barrier);
      return success();
    }

    LLVM_DEBUG(DBGS() << "barrier is necessary: " << barrier << " "
                      << barrier.getLoc() << "\n");
    return failure();
  }
};
} // namespace

void transform_dialect::EliminateGpuBarriersOp::build(OpBuilder &builder,
                                                      OperationState &state,
                                                      Value target) {
  build(builder, state, target.getType(), target);
}

DiagnosedSilenceableFailure
transform_dialect::EliminateGpuBarriersOp::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  RewritePatternSet patterns(target.getContext());
  patterns.insert<BarrierElimination>(getContext());
  ErrorCheckingTrackingListener listener(state, *this);
  auto checkErrors = llvm::make_scope_exit([&]() {
    // The TrackingListener API makes checking for errors mandatory. It is safe
    // to drop payload ops during this transform, so we can ignore all errors.
    (void)listener.checkAndResetError();
  });
  GreedyRewriteConfig config;
  config.listener = &listener;
  if (failed(
          applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
    return emitDefaultSilenceableFailure(target);
  }

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform_dialect::PackSharedMemoryAllocOp::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  packSharedMemoryAlloc(target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::PackSharedMemoryAllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

class TestVectorLayoutOptions : public VectorLayoutOptions {
public:
  TestVectorLayoutOptions(Operation *root) : VectorLayoutOptions(root) {}

  void setAnchorOps(VectorLayoutAnalysis &analysis) override {
    setAnchorOpsFromAttributes(analysis, root);
  }
};

DiagnosedSilenceableFailure
transform_dialect::TestAMDGPUContractionDistribution::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  TestVectorLayoutOptions options(target);
  RewritePatternSet patterns(target.getContext());
  populateAMDGPUDistributionPatterns(patterns);
  distributeVectorOps(target, patterns, options);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::TestAMDGPUContractionDistribution::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.cpp.inc"
