// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-expand-gpu-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EXPANDGPUOPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Clone the combiner region body, mapping block arguments to (lhs, rhs),
/// and return the value produced by the yield.
static Value inlineCombiner(RewriterBase &rewriter, Location loc,
                            Region &combiner, Value lhs, Value rhs) {
  Block &block = combiner.front();
  IRMapping mapping;
  mapping.map(block.getArgument(0), lhs);
  mapping.map(block.getArgument(1), rhs);
  for (auto &op : block.without_terminator()) {
    rewriter.clone(op, mapping);
  }
  auto yield = cast<IREE::GPU::YieldOp>(block.getTerminator());
  return mapping.lookupOrDefault(yield.getOperand(0));
}

/// Lower `iree_gpu.subgroup_scan` to a Hillis-Steele scan using
/// `gpu.shuffle idx`.
/// Computes an inclusive scan via Hillis-Steele. For inclusive mode, the
/// result is used directly. For exclusive mode, derives the result by
/// shifting right by one cluster position and inserting the identity at
/// position 0. Also computes the cluster total by shuffling from the last
/// lane in each cluster.
///
struct LowerSubgroupScan : OpRewritePattern<IREE::GPU::SubgroupScanOp> {
  LowerSubgroupScan(MLIRContext *ctx, unsigned subgroupSize,
                    PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(IREE::GPU::SubgroupScanOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value val = op.getValue();
    uint32_t clusterSize = op.getClusterSize().value_or(subgroupSize);
    uint32_t clusterStride = op.getClusterStride();
    Type valTy = val.getType();
    unsigned bitwidth = valTy.getIntOrFloatBitWidth();

    constexpr unsigned shuffleBitwidth = 32;

    if (!valTy.isIntOrFloat() || bitwidth > shuffleBitwidth) {
      return rewriter.notifyMatchFailure(
          op, "value type is not a compatible scalar");
    }

    // Validate that the cluster fits within the subgroup.
    if (clusterSize * clusterStride > subgroupSize) {
      return op.emitOpError()
             << "cluster size (" << clusterSize << ") * stride ("
             << clusterStride << ") exceeds subgroup size " << subgroupSize;
    }

    // Pack/unpack functions for types narrower than 32 bits.
    auto shuffleIntType = rewriter.getIntegerType(shuffleBitwidth);
    auto equivIntType = rewriter.getIntegerType(bitwidth);

    auto packFn = [&](Value v) -> Value {
      if (bitwidth == shuffleBitwidth) {
        return v;
      }
      Value asInt = arith::BitcastOp::create(rewriter, loc, equivIntType, v);
      return arith::ExtUIOp::create(rewriter, loc, shuffleIntType, asInt);
    };
    auto unpackFn = [&](Value v) -> Value {
      if (bitwidth == shuffleBitwidth) {
        return v;
      }
      Value asInt = arith::TruncIOp::create(rewriter, loc, equivIntType, v);
      return arith::BitcastOp::create(rewriter, loc, valTy, asInt);
    };

    Value laneId =
        gpu::LaneIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);

    // Compute logical position within cluster:
    //   lanePos = (laneId / clusterStride) % clusterSize
    Value strideCst =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(clusterStride));
    Value sizeCst =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(clusterSize));
    Value laneIdI32 = arith::IndexCastOp::create(rewriter, loc,
                                                 rewriter.getI32Type(), laneId);
    // j
    Value lanePos = arith::RemUIOp::create(
        rewriter, loc,
        arith::DivUIOp::create(rewriter, loc, laneIdI32, strideCst), sizeCst);

    Value subgroupSizeCst =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(subgroupSize));

    // Hillis-Steele inclusive scan.
    // j = lane_id
    // for i from 0 to log(N)
    //   x[i+1][j] = (j < 2^i) ?
    //                   x[i][j] :
    //                   x[i][j] op x[i][j - 2^i]
    unsigned numSteps = llvm::Log2_32(clusterSize);
    // Serial loop over log(N) steps.
    for (unsigned step = 0; step < numSteps; ++step) {
      int64_t offset = static_cast<int64_t>(1U << step) * clusterStride;
      Value offsetCst =
          arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                    rewriter.getI32IntegerAttr(offset));

      // The subtraction may wrap for lanes near the start of a cluster.
      // The wrapped value will cause the shuffle to read from an arbitrary
      // lane, but the predicate below discards the result in that case.
      Value srcLane =
          arith::SubIOp::create(rewriter, loc, laneIdI32, offsetCst);

      // Shuffle to get neighbor's value. Because we support non-unit cluster
      // strides, we need to use `idx` shuffle and not `up` shuffle.
      Value packed = packFn(val);
      auto shuffle =
          gpu::ShuffleOp::create(rewriter, loc, packed, srcLane,
                                 subgroupSizeCst, gpu::ShuffleMode::IDX);
      // x[i][j - 2^i]
      Value neighbor = unpackFn(shuffle.getShuffleResult());

      // x[i][j] op x[i][j - 2^i]
      Value combined =
          inlineCombiner(rewriter, loc, op.getCombiner(), neighbor, val);

      // Predicate: only accumulate if lanePos >= (1 << step).
      Value threshold =
          arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                    rewriter.getI32IntegerAttr(1 << step));
      // j >= 2^i
      Value predicate = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::uge, lanePos, threshold);

      // x[i+1][j] = (j < 2^i) ? x[i][j] : x[i][j] op x[i][j - 2^i]
      val = arith::SelectOp::create(rewriter, loc, predicate, combined, val);
    }
    // val now holds the inclusive scan result.

    // Compute total: shuffle from the last lane in each cluster.
    //   baseLane = laneIdI32 - lanePos * clusterStride
    //   lastLane = baseLane + (clusterSize - 1) * clusterStride
    Value lastOffset = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr((clusterSize - 1) * clusterStride));
    Value posTimesStride =
        arith::MulIOp::create(rewriter, loc, lanePos, strideCst);
    Value baseLane =
        arith::SubIOp::create(rewriter, loc, laneIdI32, posTimesStride);
    Value lastLane = arith::AddIOp::create(rewriter, loc, baseLane, lastOffset);
    Value packedForTotal = packFn(val);
    auto totalShuffle =
        gpu::ShuffleOp::create(rewriter, loc, packedForTotal, lastLane,
                               subgroupSizeCst, gpu::ShuffleMode::IDX);
    Value total = unpackFn(totalShuffle.getShuffleResult());

    if (op.getInclusive()) {
      // Inclusive: use the Hillis-Steele result directly.
      rewriter.replaceOp(op, {val, total});
      return success();
    }

    // Derive exclusive result: shift inclusive result right by one cluster
    // position, insert identity at position 0.
    //   predLane = laneIdI32 - clusterStride
    //   shifted = shuffle idx val, predLane
    //   scanResult = (lanePos == 0) ? identity : shifted
    Value scanResult;
    Value identity = op.getIdentity();
    Value predLane = arith::SubIOp::create(rewriter, loc, laneIdI32, strideCst);
    Value packedForShift = packFn(val);
    auto shiftShuffle =
        gpu::ShuffleOp::create(rewriter, loc, packedForShift, predLane,
                               subgroupSizeCst, gpu::ShuffleMode::IDX);
    Value shifted = unpackFn(shiftShuffle.getShuffleResult());

    Value zero = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                           rewriter.getI32IntegerAttr(0));
    Value isFirstLane = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, lanePos, zero);
    scanResult =
        arith::SelectOp::create(rewriter, loc, isFirstLane, identity, shifted);

    rewriter.replaceOp(op, {scanResult, total});
    return success();
  }

private:
  unsigned subgroupSize;
};

struct ExpandGPUOpsPass final : impl::ExpandGPUOpsPassBase<ExpandGPUOpsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp->emitOpError("missing subgroup size");
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    auto execTarget = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    StringRef targetArch = target.getArch();
    auto maybeChipset = amdgpu::Chipset::parse(targetArch);
    if (succeeded(maybeChipset) && isROCMBackend(execTarget)) {
      populateGpuLowerSubgroupReduceToDPPPatterns(
          patterns, *subgroupSize, *maybeChipset, PatternBenefit(2));
      populateGpuLowerClusteredSubgroupReduceToDPPPatterns(
          patterns, *subgroupSize, *maybeChipset, PatternBenefit(2));
    }

    populateGpuBreakDownSubgroupReducePatterns(
        patterns, /* maxShuffleBitwidth=*/32, PatternBenefit(3));
    populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
        patterns, *subgroupSize, /* shuffleBitwidth=*/32, PatternBenefit(1));
    patterns.add<LowerSubgroupScan>(ctx, *subgroupSize, PatternBenefit(1));
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  };
};

} // namespace

} // namespace mlir::iree_compiler
