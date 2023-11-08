// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vector-reduction-to-gpu"

namespace mlir {
namespace iree_compiler {

void debugPrint(func::FuncOp funcOp, const char *message) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << message << " ---//\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
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

/// Returns true if the given op is a memref.load from a uniform buffer or
/// read-only storage buffer.
static bool isUniformLoad(Operation *op) {
  using namespace IREE::HAL;

  auto loadOp = dyn_cast<memref::LoadOp>(op);
  if (!loadOp)
    return false;
  auto space = loadOp.getMemRefType().getMemorySpace();
  auto attr = llvm::dyn_cast_if_present<DescriptorTypeAttr>(space);
  if (!attr)
    return false;

  if (attr.getValue() == DescriptorType::UniformBuffer)
    return true;

  auto subspan = loadOp.getMemRef().getDefiningOp<InterfaceBindingSubspanOp>();
  if (!subspan)
    return false;
  if (auto flags = subspan.getDescriptorFlags()) {
    if (bitEnumContainsAll(*flags, IREE::HAL::DescriptorFlags::ReadOnly))
      return true;
  }
  return false;
}

/// Hoist uniform operations as well as special hal operations that have side
/// effect but are safe to move out of the warp single lane region.
static void
moveScalarAndBindingUniformCode(vector::WarpExecuteOnLane0Op warpOp) {
  /// Hoist ops without side effect as well as special binding ops.
  auto canBeHoisted = [](Operation *op,
                         function_ref<bool(Value)> definedOutside) {
    if (op->getNumRegions() != 0)
      return false;
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;
    if (isMemoryEffectFree(op))
      return true;

    if (isa<IREE::HAL::InterfaceBindingSubspanOp,
            IREE::HAL::InterfaceConstantLoadOp, memref::AssumeAlignmentOp>(op))
      return true;
    if (isUniformLoad(op))
      return true;
    // Shared memory is already scoped to the workgroup and can safely be
    // hoisted out of the the warp op.
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      if (hasSharedMemoryAddressSpace(allocOp.getType())) {
        return true;
      }
    }

    return false;
  };
  Block *body = warpOp.getBody();

  // Keep track of the ops we want to hoist.
  llvm::SmallSetVector<Operation *, 8> opsToMove;

  // Helper to check if a value is or will be defined outside of the region.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto *definingOp = value.getDefiningOp();
    return (definingOp && opsToMove.count(definingOp)) ||
           warpOp.isDefinedOutsideOfRegion(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there.
  for (auto &op : body->without_terminator()) {
    bool hasVectorResult = llvm::any_of(op.getResults(), [](Value result) {
      return llvm::isa<VectorType>(result.getType());
    });
    if ((!hasVectorResult || isUniformLoad(&op)) &&
        canBeHoisted(&op, isDefinedOutsideOfBody)) {
      opsToMove.insert(&op);
    }
  }

  // Move all the ops marked as uniform outside of the region.
  for (Operation *op : opsToMove)
    op->moveBefore(warpOp);
}

namespace {

/// Pattern to convert InsertElement to broadcast, this is a workaround until
/// MultiDimReduction distribution is supported.
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

/// Pattern to sink `gpu.barrier` ops out of a `warp_execute_on_lane_0` op.
class WarpOpBarrier : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<vector::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    Operation *lastNode = yield->getPrevNode();
    auto barrierOp = dyn_cast_or_null<gpu::BarrierOp>(lastNode);
    if (!barrierOp)
      return failure();

    rewriter.setInsertionPointAfter(warpOp);
    (void)rewriter.create<gpu::BarrierOp>(barrierOp.getLoc());
    rewriter.eraseOp(barrierOp);
    return success();
  }
};

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

class VectorReductionToGPUPass
    : public VectorReductionToGPUBase<VectorReductionToGPUPass> {
public:
  explicit VectorReductionToGPUPass(
      std::function<int(func::FuncOp)> getWarpSize)
      : getWarpSize(getWarpSize) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, gpu::GPUDialect,
                    affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Pre-process multiDimReductions.
    // TODO: Remove once MultiDimReduce is supported by distribute patterns.
    {
      RewritePatternSet patterns(ctx);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      // Add clean up patterns after lowering of multidimreduce lowering.
      patterns.add<InsertElementToBroadcast>(ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #1: preprocessing reduction ops");

    auto workgroupSize = llvm::map_to_vector(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return llvm::cast<IntegerAttr>(attr).getInt(); });
    assert(workgroupSize[1] == 1 && workgroupSize[2] == 1);
    // 2. Create the warp op and move the function body into it.
    const int groupSize = workgroupSize[0];
    Location loc = funcOp.getLoc();
    OpBuilder builder(funcOp);
    auto threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                   gpu::Dimension::x);
    auto cstGroupSize = builder.create<arith::ConstantIndexOp>(loc, groupSize);
    auto warpOp = builder.create<vector::WarpExecuteOnLane0Op>(
        loc, TypeRange(), threadX.getResult(), groupSize);
    warpOp.getWarpRegion().takeBody(funcOp.getFunctionBody());
    Block &newBlock = funcOp.getFunctionBody().emplaceBlock();
    threadX->moveBefore(&newBlock, newBlock.end());
    cstGroupSize->moveBefore(&newBlock, newBlock.end());
    warpOp->moveBefore(&newBlock, newBlock.end());
    warpOp.getWarpRegion().getBlocks().back().back().moveBefore(&newBlock,
                                                                newBlock.end());
    builder.setInsertionPointToEnd(&warpOp.getWarpRegion().getBlocks().back());
    builder.create<vector::YieldOp>(loc);

    debugPrint(funcOp, "after step #2: wrapping code with the warp execute op");

    // 3. Hoist the scalar code outside of the warp region.
    moveScalarAndBindingUniformCode(warpOp);

    debugPrint(funcOp, "after step #3: hosting uniform code");

    // 4. Distribute transfer write operations and propagate vector
    // distribution.
    {
      int warpSize = this->getWarpSize ? this->getWarpSize(funcOp) : 32;
      auto groupReductionFn = [&](Location loc, OpBuilder &builder, Value input,
                                  vector::CombiningKind kind, uint32_t size) {
        return emitGPUGroupReduction(loc, builder, input, kind, size, warpSize);
      };
      auto distributionFn = [](Value val) {
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
      };
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionFn, simpleWarpShuffleFunction);
      vector::populateDistributeReduction(patterns, groupReductionFn);
      vector::populateDistributeTransferWriteOpPatterns(patterns,
                                                        distributionFn);
      patterns.add<WarpOpBarrier>(patterns.getContext(), 3);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #4: propagating distribution");

    // 5. Lower the remaining WarpExecuteOnLane0 ops.
    {
      RewritePatternSet patterns(ctx);
      vector::WarpExecuteOnLane0LoweringOptions options;
      options.warpAllocationFn = allocateGlobalSharedMemory;
      options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp) {
        builder.create<gpu::BarrierOp>(loc);
      };
      vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #5: lowering remaing ops");
  }

private:
  std::function<int(func::FuncOp)> getWarpSize;
};

} // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertVectorReductionToGPUPass(
    std::function<int(func::FuncOp)> getWarpSize) {
  return std::make_unique<VectorReductionToGPUPass>(getWarpSize);
}

} // namespace iree_compiler
} // namespace mlir
