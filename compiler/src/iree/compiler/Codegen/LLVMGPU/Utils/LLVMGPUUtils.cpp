// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;

#define DEBUG_TYPE "llvm-gpu-utils"

namespace mlir::iree_compiler {

using IREE::VectorExt::NestedLayoutAttr;
using IREE::VectorExt::ToLayoutOp;
using IREE::VectorExt::VectorLayoutInterface;

static bool isContiguousStore(Operation *write) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(write)) {
    if (!transferWrite.getPermutationMap().isMinorIdentity() ||
        !transferWrite.isDimInBounds(0) || transferWrite.getMask()) {
      LDBG() << "--not a contiguous store op: " << *write;
      return false;
    }
    return true;
  }
  if (isa<vector::StoreOp>(write)) {
    return true;
  }
  LDBG() << "--not a store op: " << write->getName().getStringRef();
  return false;
}

static bool isContiguousRead(Operation *read) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(read)) {
    if (!transferRead.isDimInBounds(0) ||
        !transferRead.getPermutationMap().isMinorIdentity()) {
      LDBG() << "--not a contiguous load op: " << *read;
      return false;
    }
    return true;
  }
  if (isa<vector::LoadOp>(read)) {
    return true;
  }
  LDBG() << "--not a load op: " << read->getName().getStringRef();
  return false;
}

static Value getMemrefOperand(Operation *op) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
    return transferWrite.getBase();
  }
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
    return transferRead.getBase();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
    return storeOp.getBase();
  }
  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return loadOp.getBase();
  }
  return Value();
}

struct MaskResult {
  vector::CreateMaskOp maskOp;
  vector::ExtractOp maybeExtractOp;
};
static MaskResult getMask(Operation *op) {
  auto transferRead = dyn_cast<vector::TransferReadOp>(op);
  if (!transferRead || !transferRead.getMask())
    return MaskResult{};
  vector::ExtractOp maybeExtractOp =
      transferRead.getMask().getDefiningOp<vector::ExtractOp>();
  auto maskOp =
      maybeExtractOp
          ? maybeExtractOp.getVector().getDefiningOp<vector::CreateMaskOp>()
          : transferRead.getMask().getDefiningOp<vector::CreateMaskOp>();
  if (maybeExtractOp) {
    if (maybeExtractOp.getStaticPosition().size() + 1 !=
        llvm::cast<VectorType>(maskOp->getResultTypes().front()).getRank()) {
      LDBG() << "----mask through extract unexpected position size -> Skip: "
             << maybeExtractOp;
      return MaskResult{};
    }
    if (maybeExtractOp.getStaticPosition().size() != 1) {
      LDBG()
          << "----only mask through 2-D -> 1-D extract supported atm -> Skip: "
          << maybeExtractOp;
      return MaskResult{};
    }
    LDBG() << "----mask through extract: " << maybeExtractOp;
  }
  return MaskResult{maskOp, maybeExtractOp};
}

static Value getMaskValue(RewriterBase &rewriter, Operation *op) {
  MaskResult maskResult = getMask(op);
  if (!maskResult.maskOp)
    return Value();
  Value count = maskResult.maskOp->getOperands().back();
  vector::ExtractOp maybeExtractOp = maskResult.maybeExtractOp;
  if (maybeExtractOp) {
    assert(maybeExtractOp.getStaticPosition().size() == 1 &&
           "expected single pos");
    int64_t sliceNum = maybeExtractOp.getStaticPosition()[0];
    // TODO: to support >2-D mask + extract, and all the cmp.
    Location loc = op->getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cmp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt,
        rewriter.create<arith::ConstantIndexOp>(loc, sliceNum),
        maskResult.maskOp->getOperands().front());
    count = rewriter.create<arith::SelectOp>(loc, cmp, count, zero);
  }
  return count;
}

static Value getValueStored(Operation *writeOp) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(writeOp)) {
    return transferWrite.getValue();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(writeOp)) {
    return storeOp.getValueToStore();
  }
  return Value();
}

static Operation::operand_range getIndices(Operation *op) {
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  llvm_unreachable("unsupported op type");
}

/// Return `true` if the conversion to async copy is legal.
static bool resultsInSupportedAsyncCopy(MemRefType memrefType,
                                        Operation::operand_range indices,
                                        VectorType vecType) {
  constexpr int64_t kSupportedCpAsyncAlignmentsInBytes[3] = {4, 8, 16};
  // Condition 1: the vectory rank must be supported.
  if (vecType.hasRank() != 1) {
    LDBG() << "----> cp.async failed, not a 1-D vector: " << vecType;
    return false;
  }

  // Condition 2: the copy size must be supported.
  bool supportedCopySize = false;
  int64_t numElements = vecType.getNumElements();
  Type elementType = vecType.getElementType();
  for (int64_t alignmentInBytes : kSupportedCpAsyncAlignmentsInBytes) {
    if (alignmentInBytes * 8 ==
        numElements * elementType.getIntOrFloatBitWidth()) {
      supportedCopySize = true;
      break;
    }
  }
  if (!supportedCopySize) {
    LDBG() << "----> cp.async alignment failed, " << numElements << " elts * "
           << elementType.getIntOrFloatBitWidth()
           << "b/elem = " << numElements * elementType.getIntOrFloatBitWidth()
           << "b is not supported by cp.async";
    return false;
  }

  // TODO: Condition 3: the alignments must be supported. For cp.async the
  // NVIDIA doc (section 6.4.1) says: "The address must be naturally aligned to
  // a multiple of the access size. If an address is not properly aligned, the
  // resulting behavior is undefined.".
  return true;
}

void createAsyncGroups(RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
                       bool useMMASync) {
  LDBG() << "Start asyncGroups: useMMASync=" << useMMASync;
  llvm::SmallSetVector<Operation *, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](Operation *writeOp) {
    if (!isContiguousStore(writeOp))
      return WalkResult::advance();
    LDBG() << "--candidate writeOp: " << *writeOp;
    Value vectorVal = getValueStored(writeOp);
    if (llvm::cast<VectorType>(vectorVal.getType()).getRank() != 1) {
      LDBG() << "----writeOp is not an inbounds 1-D minor identity -> Skip";
      return WalkResult::advance();
    }
    Value memrefOperand = getMemrefOperand(writeOp);
    if (!hasSharedMemoryAddressSpace(
            llvm::cast<MemRefType>(memrefOperand.getType()))) {
      LDBG() << "----address space is not workgroup -> Skip";
      return WalkResult::advance();
    }
    Operation *readOp = vectorVal.getDefiningOp();
    if (readOp == nullptr || !isContiguousRead(readOp)) {
      LDBG() << "----no contiguous readOp defining the writeOp -> Skip";
      return WalkResult::advance();
    }

    LDBG() << "--candidate readOp: " << *readOp;
    if (auto transferRead = dyn_cast<vector::TransferReadOp>(readOp)) {
      if (transferRead.getMask()) {
        auto paddingCst =
            transferRead.getPadding().getDefiningOp<arith::ConstantFloatOp>();
        if (!paddingCst || !paddingCst.value().isZero()) {
          LDBG() << "----read padding value is not 0.f -> Skip";
          return WalkResult::advance();
        }
        auto maskResult = getMask(transferRead);
        if (!maskResult.maskOp) {
          LDBG() << "----read mask is not a vector.create_mask op -> Skip: "
                 << transferRead.getMask();
          return WalkResult::advance();
        }
      }
    }

    // Check whether both accesses are supported before we emit: this is
    // necessary to ensure the correctness of DeviceAsyncCopyOp.
    VectorType vecType = llvm::cast<VectorType>(vectorVal.getType());
    Value storeBase = getMemrefOperand(writeOp);
    Value loadBase = getMemrefOperand(readOp);
    if (!resultsInSupportedAsyncCopy(cast<MemRefType>(loadBase.getType()),
                                     getIndices(readOp), vecType) ||
        !resultsInSupportedAsyncCopy(cast<MemRefType>(storeBase.getType()),
                                     getIndices(writeOp), vecType))
      return WalkResult::advance();

    LDBG() << "--writeOp can be made async -> SUCCESS";
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<Operation *> group;
    Operation *writeOp = *copyToSharedMem.begin();
    LDBG() << "--START a group from: " << *writeOp;
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation *nextNode = writeOp;
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      // ignore read from a different address space.
      if (isa<vector::TransferReadOp, vector::LoadOp>(nextNode)) {
        Operation *readOp = nextNode;
        Value memrefOperand = getMemrefOperand(readOp);
        if (!hasSharedMemoryAddressSpace(
                llvm::cast<MemRefType>(memrefOperand.getType()))) {
          continue;
        }
      }
      if (copyToSharedMem.count(nextNode)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextNode);
        group.push_back(nextNode);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      LDBG() << "----> STOP accumulating into group due to: " << *nextNode;
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    for (Operation *writeOp : group) {
      rewriter.setInsertionPoint(writeOp);
      Value vectorVal = getValueStored(writeOp);
      auto vectorType = llvm::cast<VectorType>(vectorVal.getType());
      int64_t numElements = vectorType.getNumElements();
      Operation *readOp = vectorVal.getDefiningOp();
      Value storeBase = getMemrefOperand(writeOp);
      Value loadBase = getMemrefOperand(readOp);
      Value mask = getMaskValue(rewriter, readOp);
      auto dstMemref = llvm::cast<MemRefType>(storeBase.getType());
      int64_t sizeInBytes =
          (dstMemref.getElementTypeBitWidth() * numElements) / 8;
      UnitAttr bypassL1 =
          useMMASync && sizeInBytes == 16 ? rewriter.getUnitAttr() : UnitAttr();
      Value token = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp->getLoc(),
          nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()), storeBase,
          getIndices(writeOp), loadBase, getIndices(readOp),
          rewriter.getIndexAttr(numElements), mask,
          /*bypassL1=*/bypassL1);
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    Value groupToken = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        tokens);
    rewriter.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                              nullptr);
    // Clean up old stores.
    for (Operation *writeOp : group)
      rewriter.eraseOp(writeOp);
  }
}

void reorderTranspose(RewriterBase &rewriter,
                      mlir::FunctionOpInterface funcOp) {
  SmallVector<vector::TransposeOp> transposeOps;
  funcOp.walk([&](Operation *op) {
    if (auto transposeOp = dyn_cast<vector::TransposeOp>(op)) {
      Operation *definingOp = transposeOp.getVector().getDefiningOp();
      if (OpTrait::hasElementwiseMappableTraits(definingOp)) {
        transposeOps.push_back(transposeOp);
      }
    }
    return WalkResult::advance();
  });

  for (auto transposeOp : transposeOps) {
    OpBuilder::InsertionGuard g(rewriter);
    Operation *op = transposeOp.getVector().getDefiningOp();
    rewriter.setInsertionPoint(op);
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    SmallVector<Value> transposedOperands;
    for (auto operand : op->getOperands()) {
      Value transposed =
          rewriter.create<vector::TransposeOp>(op->getLoc(), operand, perm);
      transposedOperands.push_back(transposed);
    }
    SmallVector<Type> resultTypes{transposedOperands.front().getType()};
    Operation *newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        transposedOperands, resultTypes, op->getAttrs());
    rewriter.replaceAllUsesWith(transposeOp.getResult(), newOp->getResult(0));
  }
}

void addBarrier(mlir::FunctionOpInterface funcOp, Operation *alloc,
                ArrayRef<Operation *> aliasGroup, bool hasAsyncCopies) {
  Block *entryBlock = &(*funcOp.getBlocks().begin());
  bool needBarrier = false;
  if (alloc->getBlock() != entryBlock) {
    needBarrier = true;
  } else {
    for (Operation &op : entryBlock->getOperations()) {
      if (&op == alloc)
        break;
      if (op.getNumRegions() != 0) {
        needBarrier = true;
        break;
      }
      if (isa<memref::AllocOp>(&op) && !llvm::is_contained(aliasGroup, &op)) {
        needBarrier = true;
        break;
      }
    }
  }
  if (!needBarrier)
    return;
  OpBuilder builder(alloc);
  // TODO: make it a option if needed.
  if (hasAsyncCopies) {
    Value groupToken = builder.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        SmallVector<Value>());
    builder.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                             builder.getI32IntegerAttr(0));
  }
  builder.create<gpu::BarrierOp>(alloc->getLoc());
}

void packSharedMemoryAlloc(mlir::FunctionOpInterface funcOp) {
  DominanceInfo dominators(funcOp);
  SmallVector<Operation *> allocs;
  funcOp.walk([&](memref::AllocOp alloc) {
    if (hasSharedMemoryAddressSpace(alloc.getType())) {
      allocs.push_back(alloc);
    }
  });
  // First sink the alloc as low as possible in the CFG.
  sinkOpsInCFG(allocs, dominators);
  SmallVector<AliasGroup> aliasGroups;
  analyseAllocsForPacking(funcOp, allocs, aliasGroups);
  // If there is 1 or less alias group there is nothing to do.
  if (aliasGroups.size() <= 1)
    return;

  // Pack all the allocations into one i8 alloc.
  // We may need to add extra barriers to make sure we are done writting or
  // reading from the previous alias group before starting a new one.
  for (size_t i = 0; i < aliasGroups.size(); i++) {
    for (Operation *alloc : aliasGroups[i]) {
      addBarrier(funcOp, alloc, aliasGroups[i]);
    }
  }

  OpBuilder builder(funcOp.getContext());
  packAllocs(builder, funcOp, aliasGroups);
}

/// Gets a unit vector of the given rank, but fills in the given dimensions
/// from the 2 element array |counts|. |dim0| is the position in the returned
/// vector to put the first element of |counts|, and |dim1| is the position to
/// put the second element. For example,
///
/// rank = 3, counts = [5, 7], dim0 = 2, dim1 = 1
/// returns [1, 5, 7]
static SmallVector<int64_t> getUnitOfRankWithDims(int64_t rank,
                                                  ArrayRef<int64_t> counts,
                                                  int64_t dim0, int64_t dim1) {
  assert(counts.size() == 2 &&
         "Unexpected non-rank 2 single subgroup dimension counts");
  SmallVector<int64_t> res(rank, 1);
  res[dim0] = counts[0];
  res[dim1] = counts[1];
  return res;
}

/// Constructs the nested layout given the layout for a single subgroup and the
/// subgroup/batch counts and orders, as well as the dimensions along which to
/// distribute the intrinsic's layout.
///
/// |outerDim| and |innerDim| refer to which dimensions are the outermost and
/// innermost for a canonical MK_KN_MN matrix multiply, for a particular
/// fragment. For example, for the B matrix of an MK_NK_MN matrix multiply,
/// we would have:
///   outerDim = 1 for the K dim
///   innerDim = 0 for the N dim
///
/// For something like MK_NKN_MN with multiple N dims, it would typically be:
///   outerDim = 1 for K
///   innerDim = 2 for the second N dim
///
/// Importantly these two dimensions always refer to the actual dimension
/// positions in the undistributed vector. For each fragment, this means:
///   A: [outerDim, innerDim] = [innerMostMDim, innerMostKDim]
///   B: [outerDim, innerDim] = [innerMostKDim, innerMostNDim]
///   C: [outerDim, innerDim] = [innerMostMDim, innerMostNDim]
///
/// And here inner most is referential to the iteration order, not the order
/// they appear per fragment (because there is no relationship between the
/// dimension order of M in A and in C, for example).
static NestedLayoutAttr createNestedLayout(
    MLIRContext *context, int64_t rank, int64_t outerDim, int64_t innerDim,
    ArrayRef<int64_t> subgroupSizes, ArrayRef<int64_t> subgroupStrides,
    ArrayRef<int64_t> batchCount, IREE::GPU::MMASingleSubgroupLayout counts) {

  LLVM_DEBUG({
    llvm::dbgs() << "Creating Nested Layout for::";
    llvm::dbgs() << "\n    outerDim = " << outerDim;
    llvm::dbgs() << "\n    innerDim = " << innerDim;
    llvm::dbgs() << "\n    subgroupSizes: ";
    llvm::interleaveComma(subgroupSizes, llvm::dbgs());
    llvm::dbgs() << "\n    subgroupStrides: ";
    llvm::interleaveComma(subgroupStrides, llvm::dbgs());
    llvm::dbgs() << "\n    batchCount: ";
    llvm::interleaveComma(batchCount, llvm::dbgs());
    llvm::dbgs() << "\n    counts.outer: ";
    llvm::interleaveComma(counts.outer, llvm::dbgs());
    llvm::dbgs() << "\n    counts.thread: ";
    llvm::interleaveComma(counts.thread, llvm::dbgs());
    llvm::dbgs() << "\n    counts.element: ";
    llvm::interleaveComma(counts.element, llvm::dbgs());
    llvm::dbgs() << "\n    counts.tstrides: ";
    llvm::interleaveComma(counts.tstrides, llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  SmallVector<int64_t> outerCount =
      getUnitOfRankWithDims(rank, counts.outer, outerDim, innerDim);
  SmallVector<int64_t> threadCount =
      getUnitOfRankWithDims(rank, counts.thread, outerDim, innerDim);
  SmallVector<int64_t> threadStrides =
      getUnitOfRankWithDims(rank, counts.tstrides, outerDim, innerDim);
  SmallVector<int64_t> elementCount =
      getUnitOfRankWithDims(rank, counts.element, outerDim, innerDim);

  auto layoutAttr = NestedLayoutAttr::get(context, subgroupSizes, batchCount,
                                          outerCount, threadCount, elementCount,
                                          subgroupStrides, threadStrides);
  return layoutAttr;
}

template <typename ContractOpTy>
static FailureOr<std::tuple<IREE::VectorExt::VectorLayoutInterface,
                            IREE::VectorExt::VectorLayoutInterface,
                            IREE::VectorExt::VectorLayoutInterface>>
getContractionLayoutImpl(IREE::GPU::MMAScheduleAttr schedule,
                         VectorContractOpInfo &opInfo,
                         ContractOpTy contractOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "Getting mma layouts for:\n" << contractOp << "\n";
    llvm::dbgs() << "For schedule: " << schedule << "\n";
  });

  int64_t rank = contractOp.getIteratorTypesArray().size();
  auto mmaAttr =
      llvm::dyn_cast<IREE::GPU::MmaInterfaceAttr>(schedule.getIntrinsic());
  if (!mmaAttr) {
    return failure();
  }

  MLIRContext *context = schedule.getContext();

  SmallVector<int64_t> bounds = contractOp.getStaticLoopRanges();
  if (llvm::any_of(bounds,
                   [](int64_t x) { return x == ShapedType::kDynamic; })) {
    return failure();
  }

  if (!llvm::all_of(opInfo.getBatchDims(),
                    [&bounds](int64_t dim) { return bounds[dim] == 1; })) {
    LLVM_DEBUG({ llvm::dbgs() << "non-unit batch dimension\n"; });
    return failure();
  }

  // Get the concrete nested layout for each matrix. Note that the struct
  // MMASingleSubgroupLayout contains the partial layout for the
  // canonical (M, K) x (K, N) -> (M, N) matmul form; while the specific
  // contract op we are looking at right now may not be exactly in that form.
  // So here we need to permute/transpose the canonical layout to match with
  // the concrete contract op.

  // Note that no matter how we permute/transpose the input contraction
  // problem, the way we view the hardware warps remain the same--that is,
  // from the hardware's perspective, a single warp has the same warp ID no
  // matter what part of the contraction it works on. Similarly here, we are
  // delinearizing the linearized GPU hardware lane ID into a n-D concatenated
  // logical warp+thread using the subgroup/thread basis, so the subgroup
  // basis should remain the same for all A/B/C matrix.

  auto [intrinsicM, intrinsicN, intrinsicK] = mmaAttr.getMNKShape();

  SmallVector<int64_t, 2> subgroupMBasis;
  SmallVector<int64_t, 2> batchMSizes;
  int64_t currMCount = schedule.getSubgroupMCount();

  auto divideGreedily = [](int64_t availableSubgroups, int64_t dimSize,
                           int64_t minDimSize) -> std::pair<int64_t, int64_t> {
    int64_t dividableDim = dimSize / minDimSize;
    int64_t subgroupsUsed = std::gcd(availableSubgroups, dividableDim);
    dividableDim /= subgroupsUsed;
    int64_t batchesUsed = dividableDim;
    return {subgroupsUsed, batchesUsed};
  };

  // Greedily break up the M subgroup and batch counts along the "M" iteration
  // bounds. We distribute as many residual subgroups as possible per M dim,
  // and then divide the remaining along batch dims. The inner most M dim is
  // always the one used for the intrinsic, meaning for a valid schedule, the
  // computed batch counts and subgroup basis will satisfy totalMSize /
  // intrinsicM = product(batchMSizes) * product(subgroupMBasis)
  for (auto dim : opInfo.getMDims()) {
    // Get the number of subgroups and batches used for this dimension based
    // on the intrinsic size and the bound size.
    int64_t subgroupsUsed, batchesUsed;
    if (dim == opInfo.getMDims().back()) {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currMCount, bounds[dim], intrinsicM);
    } else {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currMCount, bounds[dim], 1);
    }
    subgroupMBasis.push_back(subgroupsUsed);
    batchMSizes.push_back(batchesUsed);
    // Update available subgroup count.
    currMCount /= subgroupsUsed;
  }

  SmallVector<int64_t, 2> subgroupNBasis;
  SmallVector<int64_t, 2> batchNSizes;
  int64_t currNCount = schedule.getSubgroupNCount();

  // Do the same for N dims.
  for (auto dim : opInfo.getNDims()) {
    // Get the number of subgroups and batches used for this dimension based
    // on the intrinsic size and the bound size.
    int64_t subgroupsUsed, batchesUsed;
    if (dim == opInfo.getNDims().back()) {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currNCount, bounds[dim], intrinsicN);
    } else {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currNCount, bounds[dim], 1);
    }
    subgroupNBasis.push_back(subgroupsUsed);
    batchNSizes.push_back(batchesUsed);
    // Update available subgroup count.
    currNCount /= subgroupsUsed;
  }

  SmallVector<int64_t> subgroupMStrides(subgroupMBasis.size());
  SmallVector<int64_t> subgroupNStrides(subgroupNBasis.size());

  auto mDimVec = opInfo.getMDims();
  llvm::SmallDenseSet<int64_t> mDims(mDimVec.begin(), mDimVec.end());
  auto nDimVec = opInfo.getNDims();
  llvm::SmallDenseSet<int64_t> nDims(nDimVec.begin(), nDimVec.end());
  // Because we currently require all batch dimensions to be unit, the
  // subgroup basis can be constructed from the M and N bases. To keep things
  // simple, the current heuristic is to distribute the loop dimensions from
  // outer to inner.
  int64_t currStride = 1;
  int64_t currM = subgroupMStrides.size() - 1;
  int64_t currN = subgroupNStrides.size() - 1;
  for (int64_t dim : llvm::reverse(llvm::seq<int64_t>(rank))) {
    if (mDims.contains(dim)) {
      subgroupMStrides[currM] = currStride;
      currStride *= subgroupMBasis[currM];
      currM--;
      continue;
    }

    if (nDims.contains(dim)) {
      subgroupNStrides[currN] = currStride;
      currStride *= subgroupNBasis[currN];
      currN--;
      continue;
    }
  }

  // C matrix layout
  auto [m, n] = opInfo.getResultMNIndex();
  int64_t cRank = opInfo.getCRank();

  // Get the M and N dims w.r.t. the dimensions of the C matrix. cMDims and
  // cNDims are the M and N dimensions of the C matrix in the order they are
  // iterated over in the contraction.
  SmallVector<int64_t> cMDims = opInfo.outMDims;
  SmallVector<int64_t> cNDims = opInfo.outNDims;
  SmallVector<int64_t> cBatchSizes(cRank, 1);
  SmallVector<int64_t> cSubgroupSizes(cRank, 1);
  SmallVector<int64_t> cSubgroupStrides(cRank, 0);
  for (auto [i, dim] : llvm::enumerate(cMDims)) {
    cBatchSizes[dim] = batchMSizes[i];
    cSubgroupSizes[dim] = subgroupMBasis[i];
    cSubgroupStrides[dim] = subgroupMStrides[i];
  }
  for (auto [i, dim] : llvm::enumerate(cNDims)) {
    cBatchSizes[dim] = batchNSizes[i];
    cSubgroupSizes[dim] = subgroupNBasis[i];
    cSubgroupStrides[dim] = subgroupNStrides[i];
  }

  IREE::VectorExt::NestedLayoutAttr cLayout = createNestedLayout(
      context, cRank, m, n,
      /*subgroupCount=*/cSubgroupSizes,
      /*subgroupStrides=*/cSubgroupStrides,
      /*batchCount=*/cBatchSizes,
      getSingleSubgroupLayout(mmaAttr, IREE::GPU::MMAFragment::Acc));
  LLVM_DEBUG({ llvm::dbgs() << "C layout: " << cLayout << "\n"; });

  // A matrix layout
  auto [afm, bfn] = opInfo.getOperandMNIndex();
  auto [afk, bfk] = opInfo.getOperandKIndex();

  int64_t aRank = opInfo.getARank();

  SmallVector<int64_t> aMDims = opInfo.lhsMDims;
  SmallVector<int64_t> aBatchSizes(aRank, 1);
  SmallVector<int64_t> aSubgroupSizes(aRank, 1);
  SmallVector<int64_t> aSubgroupStrides(aRank, 0);
  for (auto [i, dim] : llvm::enumerate(aMDims)) {
    aBatchSizes[dim] = batchMSizes[i];
    aSubgroupSizes[dim] = subgroupMBasis[i];
    aSubgroupStrides[dim] = subgroupMStrides[i];
  }
  for (auto [kDim, lhsKDim] :
       llvm::zip_equal(opInfo.getKDims(), opInfo.lhsKDim)) {
    aBatchSizes[lhsKDim] = bounds[kDim];
  }
  aBatchSizes[afk] = bounds[opInfo.getKDims().back()] / intrinsicK;

  IREE::VectorExt::NestedLayoutAttr aLayout = createNestedLayout(
      context, aRank, afm, afk,
      /*subgroupCount=*/aSubgroupSizes,
      /*subgroupStrides=*/aSubgroupStrides,
      /*batchCount=*/aBatchSizes,
      getSingleSubgroupLayout(mmaAttr, IREE::GPU::MMAFragment::Lhs));
  LLVM_DEBUG({ llvm::dbgs() << "A layout: " << aLayout << "\n"; });

  int64_t bRank = opInfo.getBRank();

  SmallVector<int64_t> bNDims = opInfo.rhsNDims;
  SmallVector<int64_t> bBatchSizes(bRank, 1);
  SmallVector<int64_t> bSubgroupSizes(bRank, 1);
  SmallVector<int64_t> bSubgroupStrides(bRank, 0);
  for (auto [i, dim] : llvm::enumerate(bNDims)) {
    bBatchSizes[dim] = batchNSizes[i];
    bSubgroupSizes[dim] = subgroupNBasis[i];
    bSubgroupStrides[dim] = subgroupNStrides[i];
  }
  for (auto [kDim, rhsKDim] :
       llvm::zip_equal(opInfo.getKDims(), opInfo.rhsKDim)) {
    bBatchSizes[rhsKDim] = bounds[kDim];
  }
  bBatchSizes[bfk] = bounds[opInfo.getKDims().back()] / intrinsicK;

  IREE::VectorExt::NestedLayoutAttr bLayout = createNestedLayout(
      context, bRank, bfk, bfn,
      /*subgroupCount=*/bSubgroupSizes,
      /*subgroupStrides=*/bSubgroupStrides,
      /*batchCount=*/bBatchSizes,
      getSingleSubgroupLayout(mmaAttr, IREE::GPU::MMAFragment::Rhs));
  LLVM_DEBUG({ llvm::dbgs() << "B layout: " << bLayout << "\n"; });

  std::tuple<VectorLayoutInterface, VectorLayoutInterface,
             VectorLayoutInterface>
      result = {aLayout, bLayout, cLayout};
  return result;
}

/// Template specializations
::mlir::FailureOr<::std::tuple<IREE::VectorExt::VectorLayoutInterface,
                               IREE::VectorExt::VectorLayoutInterface,
                               IREE::VectorExt::VectorLayoutInterface>>
IREE::GPU::getContractionLayout(IREE::GPU::MMAScheduleAttr scheduleAttr,
                                VectorContractOpInfo &opInfo,
                                linalg::LinalgOp contractOp) {
  return getContractionLayoutImpl(scheduleAttr, opInfo, contractOp);
}

::mlir::FailureOr<::std::tuple<IREE::VectorExt::VectorLayoutInterface,
                               IREE::VectorExt::VectorLayoutInterface,
                               IREE::VectorExt::VectorLayoutInterface>>
IREE::GPU::getContractionLayout(IREE::GPU::MMAScheduleAttr scheduleAttr,
                                VectorContractOpInfo &opInfo,
                                vector::ContractionOp contractOp) {
  return getContractionLayoutImpl(scheduleAttr, opInfo, contractOp);
}

} // namespace mlir::iree_compiler
