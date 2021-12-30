// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Utils.cpp - Utility functions used in Linalg to SPIR-V lowering ----===//
//
// Implementaiton of utility functions used while lowering from Linalg to SPIRV.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/SPIRV/Utils.h"

#include "iree/compiler/Codegen/SPIRV/MemorySpace.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

const char *getSPIRVDistributeAttrName() { return "iree.spirv.distribute_dim"; }

spirv::TargetEnvAttr getSPIRVTargetEnvAttr(Operation *op) {
  auto variant = op->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variant) return nullptr;
  IREE::HAL::ExecutableTargetAttr targetAttr = variant.target();
  if (!targetAttr) return nullptr;
  auto config = targetAttr.getConfiguration();
  if (!config) return nullptr;
  return config.getAs<spirv::TargetEnvAttr>(spirv::getTargetEnvAttrName());
}

LogicalResult updateWorkGroupSize(FuncOp funcOp,
                                  ArrayRef<int64_t> workGroupSize) {
  // Need to update both the surrounding FuncOp that has the spv.entry_point_abi
  // attribute, and the hal.executable.
  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body))
    return funcOp.emitError("unhandled dispatch function with multiple blocks");

  if (workGroupSize.size() != 3)
    return funcOp.emitError("expected workgroup size to have three entries");
  SmallVector<int32_t, 3> workGroupSizeVec = llvm::to_vector<3>(llvm::map_range(
      workGroupSize, [](int64_t v) { return static_cast<int32_t>(v); }));

  funcOp->setAttr(
      spirv::getEntryPointABIAttrName(),
      spirv::getEntryPointABIAttr(workGroupSizeVec, funcOp.getContext()));
  return success();
}

LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

Optional<Value> allocateWorkgroupMemory(OpBuilder &b, memref::SubViewOp subview,
                                        ArrayRef<Value> boundingSubViewSize,
                                        DataLayout &layout) {
  // Allocate the memory into the entry block of the parent FuncOp. This better
  // aligns with the semantics of this memory which is available at the entry of
  // the function.
  OpBuilder::InsertionGuard guard(b);
  FuncOp funcOp = subview->getParentOfType<FuncOp>();
  if (!funcOp) {
    subview.emitError("expected op to be within std.func");
    return llvm::None;
  }
  b.setInsertionPointToStart(&(*funcOp.getBody().begin()));
  // The bounding subview size is expected to be constant. This specified the
  // shape of the allocation.
  SmallVector<int64_t, 2> shape = llvm::to_vector<2>(
      llvm::map_range(boundingSubViewSize, [](Value v) -> int64_t {
        APInt value;
        if (matchPattern(v, m_ConstantInt(&value))) return value.getSExtValue();
        return -1;
      }));
  if (llvm::any_of(shape, [](int64_t v) { return v == -1; })) return {};
  MemRefType allocType = MemRefType::get(
      shape, subview.getType().getElementType(), {}, getWorkgroupMemorySpace());
  Value buffer = b.create<memref::AllocOp>(subview.getLoc(), allocType);
  return buffer;
}

LogicalResult deallocateWorkgroupMemory(OpBuilder &b, Value buffer) {
  // There is no utility of an explicit deallocation (as of now). Instead the
  // workgroup memory is effectively stack memory that is automatically dead at
  // the end of the function. The SPIR-V lowering treats such deallocs as
  // no-ops. So dont insert it in the first place, rather just check that the
  // deallocation is for workgroup memory.
  MemRefType bufferType = buffer.getType().dyn_cast<MemRefType>();
  if (!bufferType) return failure();
  return success(bufferType.getMemorySpaceAsInt() == getWorkgroupMemorySpace());
}

template <typename GPUIdOp, typename GPUCountOp>
static linalg::ProcInfo getGPUProcessorIdAndCountImpl(OpBuilder &builder,
                                                      Location loc,
                                                      unsigned dim) {
  assert(dim < kNumGPUDims && "processor index out of range!");

  std::array<const char *, kNumGPUDims> dimAttr{"x", "y", "z"};
  StringAttr attr = builder.getStringAttr(dimAttr[dim]);
  Type indexType = builder.getIndexType();
  return {builder.create<GPUIdOp>(loc, indexType, attr),
          builder.create<GPUCountOp>(loc, indexType, attr)};
}

template <>
linalg::ProcInfo getGPUProcessorIdAndCountImpl<GPUGlobalId, GPUGlobalCount>(
    OpBuilder &builder, Location loc, unsigned dim) {
  assert(dim < kNumGPUDims && "processor index out of range!");

  std::array<const char *, kNumGPUDims> dimAttr{"x", "y", "z"};
  StringAttr attr = builder.getStringAttr(dimAttr[dim]);
  Type indexType = builder.getIndexType();
  Value gridDim = builder.create<gpu::GridDimOp>(loc, indexType, attr);
  Value blockId = builder.create<gpu::BlockIdOp>(loc, indexType, attr);
  Value blockDim = builder.create<gpu::BlockDimOp>(loc, indexType, attr);
  Value threadId = builder.create<gpu::ThreadIdOp>(loc, indexType, attr);
  // TODO(ravishankarm): Using affine_maps here would be beneficial, and we can
  // do this because the blockDim is constant. But this would lead to an
  // ordering issue cause it assumes that the workgroup size has already been
  // set. If using affine_map can help, make sure that the workgroup size is set
  // before.
  return {
      builder.create<arith::AddIOp>(
          loc, builder.create<arith::MulIOp>(loc, blockId, blockDim), threadId),
      builder.create<arith::MulIOp>(loc, blockDim, gridDim)};
}

template <typename GPUIdOp, typename GPUCountOp>
static SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCountsImpl(
    OpBuilder &builder, Location loc, unsigned numDims) {
  SmallVector<linalg::ProcInfo, 2> procInfo(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] =
        getGPUProcessorIdAndCountImpl<GPUIdOp, GPUCountOp>(builder, loc, i);
  }
  return procInfo;
}

template <typename GPUIdOp, typename GPUCountOp>
SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCounts(OpBuilder &builder,
                                                             Location loc,
                                                             unsigned numDims) {
  return getGPUProcessorIdsAndCountsImpl<GPUIdOp, GPUCountOp>(builder, loc,
                                                              numDims);
}

/// Explicit instantiation of gpuGPUProcessorIdsAndCounts.
template SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCounts<gpu::BlockIdOp, gpu::GridDimOp>(OpBuilder &builder,
                                                            Location loc,
                                                            unsigned numDims);
template SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
    OpBuilder &builder, Location loc, unsigned numDims);
template SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCounts<GPUGlobalId, GPUGlobalCount>(OpBuilder &builder,
                                                         Location loc,
                                                         unsigned numDims);

scf::ParallelOp collapseParallelLoops(PatternRewriter &rewriter,
                                      scf::ParallelOp pLoopOp) {
  if (pLoopOp.getNumReductions()) return nullptr;

  unsigned numLoops = pLoopOp.getNumLoops();
  if (numLoops == 1) return pLoopOp;

  // Compute the number of iterations of each loops starting from the innermost.
  Location loc = pLoopOp.getLoc();
  Value totalNumIterations = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Track the "stride" of each loop, i.e. product of the total number of
  // iterations of the inner loops.
  SmallVector<Value, 2> iterationStride;
  iterationStride.resize(pLoopOp.getNumLoops());
  auto lbs = pLoopOp.getLowerBound();
  auto ubs = pLoopOp.getUpperBound();
  auto steps = pLoopOp.getStep();
  for (int i = numLoops - 1; i >= 0; --i) {
    Value lb = lbs[i], ub = ubs[i], step = steps[i];
    Value iterCount = rewriter.create<arith::DivSIOp>(
        loc, rewriter.create<arith::SubIOp>(loc, ub, lb), step);
    iterationStride[i] = totalNumIterations;
    totalNumIterations =
        rewriter.create<arith::MulIOp>(loc, totalNumIterations, iterCount);
  }

  // Create the collapsed parallel loop op with lowerbound 0, step 1 and upper
  // bound being the totalNumIterations.
  Value newLb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value newStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::ParallelOp newPLoopOp =
      rewriter.create<scf::ParallelOp>(loc, newLb, totalNumIterations, newStep);

  // Build the body of the collapsed loop by cloning the original loop body. The
  // replacement value of the induction variables of the original loop body,
  // from the induction variable of the new loop, using
  //   origLoopIv[i] = loopIv / iterationStride[i]
  //   loopIv = loopIv % iterationStride[i]
  OpBuilder::InsertionGuard guard(rewriter);
  Block &pLoopBody = pLoopOp.getLoopBody().front();
  rewriter.setInsertionPointToStart(&newPLoopOp.getLoopBody().front());
  Value loopIv = *newPLoopOp.getInductionVars().begin();
  BlockAndValueMapping map;
  for (int i : llvm::seq<int>(0, numLoops)) {
    Value iterNum =
        rewriter.create<arith::DivSIOp>(loc, loopIv, iterationStride[i]);
    AffineExpr d0, d1;
    bindDims(rewriter.getContext(), d0, d1);
    AffineExpr s0 = getAffineSymbolExpr(0, rewriter.getContext());
    Value newIv = makeComposedAffineApply(rewriter, loc, d0 + d1 * s0,
                                          {lbs[i], iterNum, steps[i]});
    map.map(pLoopBody.getArgument(i), newIv);
    loopIv = rewriter.create<arith::RemSIOp>(loc, loopIv, iterationStride[i]);
  }
  for (Operation &op : pLoopBody.without_terminator()) {
    rewriter.clone(op, map);
  }
  rewriter.eraseOp(pLoopOp);
  return newPLoopOp;
}

/// Builds an empty scf.for operation. The default builder adds an entry basic
/// block which needs to be avoided here.
static scf::ForOp buildEmptyForOp(Location loc, OpBuilder &builder, Value lb,
                                  Value ub, Value step) {
  OperationState state(loc, scf::ForOp::getOperationName());
  state.addOperands({lb, ub, step});
  state.addRegion();
  return cast<scf::ForOp>(builder.createOperation(state));
}

Operation *replacePLoopOp(ConversionPatternRewriter &rewriter,
                          scf::ParallelOp pLoopOp,
                          ArrayRef<LoopBounds> newPLoopBounds,
                          ArrayRef<LoopBounds> forBounds,
                          ArrayRef<unsigned> permutation) {
  assert(!forBounds.empty() && "unhandled case of no scf.for created");
  unsigned numLoops = pLoopOp.getNumLoops();
  Location loc = pLoopOp.getLoc();
  assert(forBounds.size() + newPLoopBounds.size() == numLoops &&
         "cannot drop loops when splitting scf.parallel operation");
  assert(permutation.size() == numLoops);
  OpBuilder::InsertionGuard guard(rewriter);

  // Need a signature conversion for the body of the scf.parallel operation,
  // before can it can be used as the body of the innermost loop created here.
  TypeConverter::SignatureConversion signatureConverter(numLoops);
  Operation *outermostLoop = nullptr;
  auto permuteIt = permutation.begin();

  // Create the scf.parallel operation as the outermost loop, if specified.
  if (!newPLoopBounds.empty()) {
    auto lbs = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) -> Value { return bounds.lb; }));
    auto ubs = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) { return bounds.ub; }));
    auto steps = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) { return bounds.step; }));
    auto newPLoop = rewriter.create<scf::ParallelOp>(loc, lbs, ubs, steps);
    for (auto iv : newPLoop.getInductionVars()) {
      signatureConverter.remapInput(*permuteIt, iv);
      permuteIt++;
    }
    rewriter.setInsertionPointToStart(newPLoop.getBody());
    outermostLoop = newPLoop.getOperation();
  }

  // Generate the nested scf.for operations with the bounds passed.
  for (auto it : enumerate(forBounds)) {
    Value lb = it.value().lb, ub = it.value().ub, step = it.value().step;
    if (it.index() != forBounds.size() - 1) {
      auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step);
      if (!outermostLoop) outermostLoop = forOp.getOperation();
      signatureConverter.remapInput(*permuteIt, forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    } else {
      // For the last loop, move the body of the scf.parallel op as the body of
      // the loop after signature conversion.
      auto forOp = buildEmptyForOp(loc, rewriter, lb, ub, step);
      if (!outermostLoop) outermostLoop = forOp.getOperation();
      signatureConverter.addInputs(*permuteIt, rewriter.getIndexType());
      Region &pLoopOpRegion = pLoopOp.getLoopBody();
      rewriter.applySignatureConversion(&pLoopOpRegion, signatureConverter);
      Region &forOpRegion = forOp.getLoopBody();
      rewriter.inlineRegionBefore(pLoopOpRegion, forOpRegion,
                                  forOpRegion.begin());
    }
    permuteIt++;
  }
  rewriter.eraseOp(pLoopOp);
  return outermostLoop;
}

/// Builds an empty scf.if operation without the then and else blocks.
static scf::IfOp buildEmptyIfOp(Location loc, OpBuilder &builder, Value cond) {
  OperationState state(loc, scf::IfOp::getOperationName());
  state.addOperands(cond);
  state.addRegion();
  state.addRegion();
  return cast<scf::IfOp>(builder.createOperation(state));
}

LogicalResult distributeSingleIterationPerProcessor(
    ConversionPatternRewriter &rewriter, scf::ParallelOp pLoopOp,
    ArrayRef<linalg::ProcInfo> procInfo, bool generateGuard) {
  unsigned numLoops = pLoopOp.getNumLoops();
  Location loc = pLoopOp.getLoc();
  assert(numLoops == procInfo.size() &&
         "expected as many ids as number of loops");

  auto lbs = pLoopOp.getLowerBound();
  auto step = pLoopOp.getStep();
  SmallVector<Value, 2> ivReplacements;
  for (unsigned i : llvm::seq<unsigned>(0, numLoops)) {
    Value iterValue = rewriter.create<arith::AddIOp>(
        loc, lbs[i],
        rewriter.create<arith::MulIOp>(loc, procInfo[i].procId, step[i]));
    ivReplacements.push_back(iterValue);
  }
  Region &pLoopOpRegion = pLoopOp.getLoopBody();

  if (generateGuard) {
    TypeConverter::SignatureConversion signatureConverter(numLoops);
    Value cond = nullptr;
    auto ubs = pLoopOp.getUpperBound();
    for (unsigned i : llvm::seq<unsigned>(0, numLoops)) {
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 ivReplacements[i], ubs[i]);
      cond = (cond ? rewriter.create<arith::AndIOp>(loc, cond, cmp) : cmp);
      signatureConverter.remapInput(i, ivReplacements[i]);
    }
    rewriter.applySignatureConversion(&pLoopOpRegion, signatureConverter);
    scf::IfOp ifOp = buildEmptyIfOp(loc, rewriter, cond);
    Region &ifOpRegion = ifOp.getRegion(0);
    rewriter.inlineRegionBefore(pLoopOpRegion, ifOpRegion, ifOpRegion.begin());
  } else {
    // The body of the scf.parallel needs to be moved into its parent
    // operation.
    // - Split the block just before the scf.parallel operation.
    // - Move the only block of scf.parallel before the newly created block
    //   (after signature conversion).
    // - Add branch from the original block to the moved block of the
    //   scf.parallel's region, and from the latter to the block created by the
    //   split operation.
    // - Canonicalization will fold these branches away.
    Block *destBlock = pLoopOp.getOperation()->getBlock();
    Block *remainingInst =
        rewriter.splitBlock(destBlock, Block::iterator(pLoopOp));
    Block *sourceBlock = &pLoopOpRegion.front();
    rewriter.eraseOp(sourceBlock->getTerminator());
    rewriter.mergeBlocks(&pLoopOpRegion.front(), destBlock, ivReplacements);
    rewriter.mergeBlocks(remainingInst, destBlock, {});
  }
  rewriter.eraseOp(pLoopOp);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
