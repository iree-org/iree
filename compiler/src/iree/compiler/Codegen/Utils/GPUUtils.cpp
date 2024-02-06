// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/GPUUtils.h"

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <cassert>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "iree-codegen-gpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static constexpr unsigned kShuffleBitWidth = 32;

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// GPU processor IDs and sizes
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getGPUThreadIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                         unsigned numDims,
                         llvm::ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] = {
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]),
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(workgroupSize[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getSubgroupIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                        unsigned warpSize, unsigned numDims,
                        llvm::ArrayRef<int64_t> numSubgroups) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    mlir::Value subgroupId =
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]);
    if (i == 0) {
      mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
      subgroupId = mlir::affine::makeComposedAffineApply(
          builder, loc, d0.floorDiv(builder.getAffineConstantExpr(warpSize)),
          {subgroupId});
    }
    procInfo[numDims - 1 - i] = {
        subgroupId,
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(numSubgroups[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

std::array<int64_t, 3> getWorkgroupSize(mlir::FunctionOpInterface funcOp) {
  std::array<int64_t, 3> workgroupSize;
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp =
      mlir::iree_compiler::getEntryPoint(funcOp);
  std::optional<mlir::ArrayAttr> workgroupSizeAttr =
      exportOp->getWorkgroupSize();
  assert(workgroupSizeAttr.has_value());
  for (auto [index, attr] : llvm::enumerate(workgroupSizeAttr.value())) {
    workgroupSize[index] =
        llvm::cast<mlir::IntegerAttr>(attr).getValue().getZExtValue();
  }
  return workgroupSize;
}

std::optional<int64_t> getSubgroupSize(mlir::FunctionOpInterface funcOp) {
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp =
      mlir::iree_compiler::getEntryPoint(funcOp);
  if (failed(exportOp)) {
    return std::nullopt;
  }
  if (IntegerAttr attr = exportOp->getSubgroupSizeAttr()) {
    return attr.getValue().getSExtValue();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// GPU vectorization
//===----------------------------------------------------------------------===//

bool canPerformVectorAccessUsingAllThreads(ArrayRef<int64_t> shape,
                                           int64_t threadCount,
                                           int64_t vectorSize) {
  // Verify that each dimension of the shape can be distributed on the
  // threads
  // For zero dim tensor, consider it's too small to access using all threads.
  if (shape.size() == 0)
    return false;
  int64_t threadsAvailable = threadCount;
  for (const auto &[index, dim] : llvm::enumerate(llvm::reverse(shape))) {
    int64_t numElementPerThread = index == 0 ? vectorSize : 1;
    int64_t numThreads = dim / numElementPerThread;
    if (numThreads == 0)
      return false;
    if (numThreads > threadsAvailable) {
      // If there are no enough remaining threads to distribute the current
      // dimension, try to use all remaining threads. But we still need to make
      // sure all work can be distributed to these threads evenly.
      if (numThreads % threadsAvailable != 0)
        return false;
      numThreads = threadsAvailable;
    }
    if (threadsAvailable % numThreads != 0)
      return false;
    threadsAvailable = threadsAvailable / numThreads;
    if (threadsAvailable == 1)
      break;
  }
  return threadsAvailable == 1;
}

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register. This is needed to get good performance on sm_80 target.
std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract) {
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : contract.getIndexingMapsArray()[0].getResults()) {
    dims.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

//===----------------------------------------------------------------------===//
// GPU workgroup memory
//===----------------------------------------------------------------------===//

std::optional<Value> allocateWorkgroupMemory(OpBuilder &builder,
                                             memref::SubViewOp subview,
                                             ArrayRef<Value> sizeBounds,
                                             DataLayout &) {
  OpBuilder::InsertionGuard guard(builder);

  mlir::FunctionOpInterface funcOp =
      subview->getParentOfType<mlir::FunctionOpInterface>();
  if (!funcOp)
    return std::nullopt;

  // The subview size bounds are expected to be constant; they specify the shape
  // of the allocation.
  SmallVector<int64_t, 2> shape;
  for (Value bound : sizeBounds) {
    APInt value;
    if (!matchPattern(bound, m_ConstantInt(&value)))
      return std::nullopt;
    shape.push_back(value.getSExtValue());
  }

  builder.setInsertionPoint(&funcOp.front(), funcOp.front().begin());
  auto type = MemRefType::get(
      shape, subview.getType().getElementType(), MemRefLayoutAttrInterface{},
      gpu::AddressSpaceAttr::get(builder.getContext(),
                                 gpu::GPUDialect::getWorkgroupAddressSpace()));
  Value buffer = builder.create<memref::AllocOp>(funcOp.getLoc(), type);
  return buffer;
}

LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value /*buffer*/) {
  return success();
}

LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  Operation *copyOp = b.create<memref::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

static bool propagateCopyDestIntoProducerFill(memref::CopyOp copyOp) {
  // Look for a fill Op writing into the copyOp source.
  Operation *prevOp = copyOp->getPrevNode();
  while (prevOp) {
    if (isMemoryEffectFree(prevOp)) {
      prevOp = prevOp->getPrevNode();
      continue;
    }

    auto fillOp = dyn_cast<linalg::FillOp>(prevOp);
    if (!fillOp)
      break;
    if (fillOp.output() != copyOp.getSource())
      break;
    // Move the fillOp and change the destination to the copy destination.
    fillOp->moveBefore(copyOp);
    fillOp.getOutputsMutable().assign(copyOp.getTarget());
    return true;
  }
  return false;
}

// Split input/output operand from copy from shared memory into a separate
// input.
static void insertInputValueIntoGeneric(Value source, linalg::GenericOp op) {
  SmallVector<Value> newOperands;
  SmallVector<AffineMap> maps;
  for (OpOperand *in : op.getDpsInputOperands()) {
    newOperands.push_back(in->get());
    maps.push_back(op.getMatchingIndexingMap(in));
  }
  newOperands.push_back(source);
  assert(op.getNumDpsInits() == 1);
  OpOperand *outOperand = op.getDpsInitOperand(0);
  maps.push_back(op.getMatchingIndexingMap(outOperand));
  maps.push_back(op.getMatchingIndexingMap(outOperand));
  Location loc = op.getLoc();
  SmallVector<utils::IteratorType> iterTypes(op.getNumLoops(),
                                             utils::IteratorType::parallel);
  OpBuilder builder(op);
  auto newOp = builder.create<linalg::GenericOp>(
      loc, newOperands, outOperand->get(), maps, iterTypes);
  newOp.getRegion().getBlocks().splice(newOp.getRegion().begin(),
                                       op.getRegion().getBlocks());

  Block &payload = newOp.getRegion().front();
  payload.addArgument(payload.getArguments().back().getType(), loc);
  setMarker(newOp, getCopyToWorkgroupMemoryMarker());
}

/// Propagate the shared memory copy into the consumer op if it's a fully
/// parallel linalg.generic.
static bool
propagateCopySourceIntoConsumerGeneric(memref::CopyOp copyOp,
                                       SmallVector<Operation *> &toDelete) {
  // Look for a generic Op reading the copyOp target.
  Operation *nextOp = copyOp->getNextNode();
  while (nextOp) {
    if (isMemoryEffectFree(nextOp)) {
      nextOp = nextOp->getNextNode();
      continue;
    }
    auto consumer = dyn_cast<linalg::GenericOp>(nextOp);
    if (!consumer || consumer.getNumDpsInits() != 1 ||
        !consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0))
             .isIdentity())
      break;
    if (*consumer.getOutputs().begin() != copyOp.getTarget())
      break;
    insertInputValueIntoGeneric(copyOp.getSource(), consumer);
    toDelete.push_back(consumer);
    return true;
  }
  return false;
}

/// This is needed because we are doing promotion to shared memory on buffers.
/// This is a fragile and temporary solution until we move to be able to do this
/// kind of transformations on tensors.
void propagateSharedMemoryCopy(mlir::FunctionOpInterface funcOp) {
  SmallVector<Operation *> toDelete;
  funcOp.walk([&toDelete](memref::CopyOp copyOp) {
    if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
      if (propagateCopyDestIntoProducerFill(copyOp) ||
          propagateCopySourceIntoConsumerGeneric(copyOp, toDelete))
        toDelete.push_back(copyOp.getOperation());
    }
  });
  for (Operation *op : toDelete)
    op->erase();
}

void insertBarriersAroundSharedMemoryCopy(mlir::FunctionOpInterface funcOp) {
  OpBuilder builder(funcOp.getContext());
  // Insert barriers before and after copies to workgroup memory and skip
  // insert barriers between back to back copy to workgroup memory.
  funcOp.walk([&builder](Operation *copyOp) {
    if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
      Operation *prevOp = copyOp->getPrevNode();
      if (!prevOp || !hasMarker(prevOp, getCopyToWorkgroupMemoryMarker())) {
        builder.setInsertionPoint(copyOp);
        builder.create<gpu::BarrierOp>(copyOp->getLoc());
      }
      Operation *nextOp = copyOp->getNextNode();
      if (!nextOp || !hasMarker(nextOp, getCopyToWorkgroupMemoryMarker())) {
        builder.setInsertionPointAfter(copyOp);
        builder.create<gpu::BarrierOp>(copyOp->getLoc());
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// Reduction utils
//===----------------------------------------------------------------------===//

/// Packs scalar element to its vector equivalent.
/// (i.e f16 -> vector<1xf16> and f32 -> vector<1xf32>)
static Value promoteElementToVector(Location loc, OpBuilder &builder,
                                    Value input) {
  VectorType vectorTypeBroadcast = VectorType::get({1}, input.getType());
  Value vectorInput =
      builder.create<vector::BroadcastOp>(loc, vectorTypeBroadcast, input);
  return vectorInput;
}

Value packVectorToSupportedWidth(Location loc, OpBuilder &builder,
                                 Value input) {
  LLVM_DEBUG({
    auto vecType = input.getType().cast<VectorType>();
    Type elementType = vecType.getElementType();
    assert(vecType.getDimSize(0) * elementType.getIntOrFloatBitWidth() ==
               kShuffleBitWidth &&
           "vecSize * vecBitWidth needs to packable into 32-bitwidth.");
    assert(elementType.isIntOrFloat() &&
           "Only int and float packing is supported.");
  });
  VectorType packed32Type = VectorType::get({1}, builder.getI32Type());
  Value packedInputVec =
      builder.create<vector::BitCastOp>(loc, packed32Type, input);
  Value packedInput = builder.create<vector::ExtractOp>(loc, packedInputVec, 0);
  return packedInput;
}

Value unpackToVector(Location loc, OpBuilder &builder, Value packedInput,
                     VectorType targetVecType) {
  LLVM_DEBUG({
    Type packedType = packedInput.getType();
    assert(packedType.isIntOrFloat() && "Only ints and floats are unpackable.");
    Type elementType = targetVecType.getElementType();
    assert(targetVecType.getDimSize(0) * elementType.getIntOrFloatBitWidth() ==
               packedType.getIntOrFloatBitWidth() &&
           "packed width needs to be unpackable to vecSize * vecBitWidth.");
  });
  Value packedVector = promoteElementToVector(loc, builder, packedInput);
  Value unpackedVector =
      builder.create<vector::BitCastOp>(loc, targetVecType, packedVector);
  return unpackedVector;
}

/// Emit warp reduction code sequence for a given scalar input value.
static Value warpReduction(Location loc, OpBuilder &builder, Value input,
                           vector::CombiningKind kind, uint32_t warpSize,
                           uint32_t numLaneToReduce) {
  assert(llvm::isPowerOf2_32(numLaneToReduce));
  assert((llvm::isa<IntegerType, FloatType>(input.getType())) &&
         "Input must be a scalar");
  IntegerType shuffleIntType = builder.getIntegerType(kShuffleBitWidth);
  Type origInputType = input.getType();
  const unsigned origBitWidth = origInputType.getIntOrFloatBitWidth();
  assert(origBitWidth <= kShuffleBitWidth && "Unsupported input type bitwidth");

  const bool needsPacking = kShuffleBitWidth != origBitWidth;
  IntegerType equivIntType = builder.getIntegerType(origBitWidth);

  // Always perform the shuffles over the supported scalar type. For inputs of
  // smaller bitwidth, perform packing and unpacking via the supported integer
  // type.
  auto unpack = [loc, &builder, needsPacking, equivIntType,
                 origInputType](Value packedVal) -> Value {
    if (!needsPacking)
      return packedVal;
    auto asInt = builder.create<arith::TruncIOp>(loc, equivIntType, packedVal);
    return builder.create<arith::BitcastOp>(loc, origInputType, asInt);
  };

  auto pack = [loc, &builder, needsPacking, equivIntType,
               shuffleIntType](Value unpackedVal) -> Value {
    if (!needsPacking)
      return unpackedVal;
    auto asInt =
        builder.create<arith::BitcastOp>(loc, equivIntType, unpackedVal);
    return builder.create<arith::ExtUIOp>(loc, shuffleIntType, asInt);
  };

  // Lane value always stays in the original type. We use it to perform arith
  // reductions.
  Value laneVal = input;
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < numLaneToReduce; i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, pack(laneVal), i,
                                                 /*width=*/warpSize,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    laneVal = makeArithReduction(builder, loc, kind, laneVal, unpack(shuffled));
  }
  // Broadcast the result to all the lanes.
  if (warpSize != numLaneToReduce) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, pack(laneVal), 0,
                                                 /*width=*/warpSize,
                                                 /*mode=*/gpu::ShuffleMode::IDX)
                         .getShuffleResult();
    laneVal = unpack(shuffled);
  }

  return laneVal;
}

// List of identity elements by operation.
// https://en.wikipedia.org/wiki/Identity_element
static TypedAttr getCombiningKindIdentity(OpBuilder &builder,
                                          vector::CombiningKind combiningKind,
                                          Type type) {
  switch (combiningKind) {
  case vector::CombiningKind::ADD:
    return builder.getZeroAttr(type);
  case vector::CombiningKind::MUL: {
    if (type.isIntOrIndex()) {
      return builder.getIntegerAttr(type, 1);
    }
    return builder.getFloatAttr(type, 1);
  }
  case vector::CombiningKind::MINUI:
  case vector::CombiningKind::MINSI:
    return builder.getIntegerAttr(type, std::numeric_limits<int64_t>::max());
  case vector::CombiningKind::MAXUI:
  case vector::CombiningKind::MAXSI:
    return builder.getIntegerAttr(type, std::numeric_limits<int64_t>::min());
  case vector::CombiningKind::AND:
    return builder.getIntegerAttr(type, 1);
  case vector::CombiningKind::OR:
  case vector::CombiningKind::XOR:
    return builder.getZeroAttr(type);
  case vector::CombiningKind::MINIMUMF:
  case vector::CombiningKind::MINNUMF: {
    auto posInfApFloat = APFloat::getInf(
        llvm::cast<FloatType>(type).getFloatSemantics(), /*Negative=*/false);
    return builder.getFloatAttr(type, posInfApFloat);
  }
  case vector::CombiningKind::MAXIMUMF:
  case vector::CombiningKind::MAXNUMF: {
    auto negInfApFloat = APFloat::getInf(
        llvm::cast<FloatType>(type).getFloatSemantics(), /*Negative=*/true);
    return builder.getFloatAttr(type, negInfApFloat);
  }
  }
  return TypedAttr();
}

/// Emit identity variable.
static Value getCombiningIdentityValue(Location loc, OpBuilder &builder,
                                       vector::CombiningKind kind,
                                       Type identityType) {
  auto vectorType = llvm::dyn_cast<VectorType>(identityType);
  Type elementType = identityType;
  if (vectorType) {
    elementType = vectorType.getElementType();
  }
  TypedAttr identityAttr = getCombiningKindIdentity(builder, kind, elementType);
  if (vectorType) {
    identityAttr = DenseElementsAttr::get(vectorType, identityAttr);
  }
  assert(identityAttr && "Unknown identity value for the reduction");
  Value identity =
      builder.create<arith::ConstantOp>(loc, identityType, identityAttr);
  return identity;
}

/// Return a matching GPU reduction operations.
static std::optional<gpu::AllReduceOperation>
combiningKindToAllReduce(vector::CombiningKind kind) {
  using gpu::AllReduceOperation;
  using vector::CombiningKind;

  switch (kind) {
  case CombiningKind::ADD:
    return AllReduceOperation::ADD;
  case CombiningKind::AND:
    return AllReduceOperation::AND;
  case CombiningKind::MUL:
    return AllReduceOperation::MUL;
  case CombiningKind::OR:
    return AllReduceOperation::OR;
  case CombiningKind::XOR:
    return AllReduceOperation::XOR;
  // Currently, the min/max reductions are not well-defined in the gpu dialect.
  // See https://github.com/llvm/llvm-project/issues/72354.
  default:
    break;
  }
  return std::nullopt;
}

/// Emit reduction across a group for a given input.
Value emitGPUGroupReduction(Location loc, OpBuilder &builder, Value input,
                            vector::CombiningKind kind, uint32_t size,
                            int warpSize, bool expandSubgroupReduce) {
  assert(
      size % warpSize == 0 &&
      "Group reduction only support for sizes aligned on warp size for now.");

  if (!expandSubgroupReduce && size == warpSize) {
    if (auto gpuReduceKind = combiningKindToAllReduce(kind)) {
      // Simple case -- emit `gpu.subgroup_reduce` directly.
      Value laneVal = builder.create<vector::ReductionOp>(loc, kind, input);
      return builder.create<gpu::SubgroupReduceOp>(loc, laneVal,
                                                   *gpuReduceKind);
    }
  }

  // More-involved case -- generate `gpu.shuffle` ops over i32 values (using the
  // butterfly shuffle algorithm).
  //
  // First reduce on a single thread to get per lane reduction value.
  Value laneVal = builder.create<vector::ReductionOp>(loc, kind, input);
  laneVal = warpReduction(loc, builder, laneVal, kind, warpSize, warpSize);
  // if we have more than one warp, reduce across warps.
  if (size > warpSize) {
    uint32_t numWarp = size / warpSize;
    assert(numWarp <= warpSize &&
           "Only support 1 level, need to implement recursive/loop for this "
           "case.");
    auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
        builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    MemRefType memrefType =
        MemRefType::get(numWarp, laneVal.getType(), MemRefLayoutAttrInterface{},
                        addressSpaceAttr);
    Value alloc = builder.create<memref::AllocOp>(loc, memrefType);
    Value threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                    gpu::Dimension::x);
    Value cstWarpSize = builder.create<arith::ConstantIndexOp>(loc, warpSize);
    Value warpId = builder.create<arith::DivUIOp>(loc, threadX, cstWarpSize);
    Value laneId = builder.create<arith::RemUIOp>(loc, threadX, cstWarpSize);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value lane0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                laneId, zero);
    // Store the reduction for each warp.
    SmallVector<Value> indices = {warpId};
    builder.create<scf::IfOp>(loc, lane0, [&](OpBuilder &b, Location l) {
      b.create<memref::StoreOp>(l, laneVal, alloc, indices);
      b.create<scf::YieldOp>(l, std::nullopt);
    });
    builder.create<gpu::BarrierOp>(loc);
    // Further reduce the outputs from each warps with a single warp reduce.
    Value memrefSize = builder.create<arith::ConstantIndexOp>(loc, numWarp - 1);
    Value laneIdInBounds =
        builder.create<arith::MinUIOp>(loc, laneId, memrefSize);
    Value loadVal = builder.create<memref::LoadOp>(loc, alloc, laneIdInBounds);
    Value cstNumWarp = builder.create<arith::ConstantIndexOp>(loc, numWarp);
    if (!llvm::isPowerOf2_32(numWarp)) {
      // Pad with identity element if numel < warpSize for valid warp reduction.
      Value useIdentityElement = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, laneId, cstNumWarp);
      numWarp = llvm::PowerOf2Ceil(numWarp);
      Value identity =
          getCombiningIdentityValue(loc, builder, kind, loadVal.getType());
      loadVal = builder.create<arith::SelectOp>(loc, useIdentityElement,
                                                identity, loadVal);
    }
    laneVal = warpReduction(loc, builder, loadVal, kind, warpSize, numWarp);
  }

  return laneVal;
}

std::optional<SmallVector<int64_t>> getWmmaNativeVectorSize(Operation *op) {
  // Currently hardcode the size of wmma operation. When more cases are
  // supported this should be picked based on what the backend supports.
  int64_t m = 16;
  int64_t n = 16;
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    int64_t k = contract.getLhsType().getElementType().isF16() ? 16 : 8;
    SmallVector<int64_t> nativeSize(contract.getIteratorTypes().size() - 3, 1);
    nativeSize.append({m, n, k});
    return nativeSize;
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    if (writeOp.getVectorType().getRank() < 2)
      return std::nullopt;
    SmallVector<int64_t> nativeSize(writeOp.getVectorType().getRank() - 2, 1);
    nativeSize.append({m, n});
    return nativeSize;
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    // Transfer read ops may need different shapes based on how they are being
    // used. For simplicity just match the shape used by the extract strided op.
    VectorType sliceType;
    for (Operation *users : op->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract)
        return std::nullopt;
      auto vecType = llvm::cast<VectorType>(extract.getResult().getType());
      if (sliceType && sliceType != vecType)
        return std::nullopt;
      sliceType = vecType;
    }
    return llvm::to_vector(sliceType.getShape());
  }
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      // TODO: The condition for unrolling elementwise should be restricted
      // only to operations that need unrolling (connected to the contract).
      if (vecType.getRank() < 2)
        return std::nullopt;

      // First check whether there is a slice to infer the shape from. This is
      // required for cases where the accumulator type differs from the input
      // types, in which case we will see an `arith.ext_` between the contract
      // and transfer_read which needs to be unrolled.
      VectorType sliceType;
      for (Operation *users : op->getUsers()) {
        auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
        if (!extract)
          return std::nullopt;
        auto vecType = llvm::cast<VectorType>(extract.getResult().getType());
        if (sliceType && sliceType != vecType)
          return std::nullopt;
        sliceType = vecType;
      }
      if (sliceType)
        return llvm::to_vector(sliceType.getShape());

      // Else unroll for trailing elementwise.
      SmallVector<int64_t> nativeSize(vecType.getRank() - 2, 1);
      // Map elementwise ops to the output shape.
      nativeSize.append({m, n});
      return nativeSize;
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// getMmaNativeVectorSize
//===----------------------------------------------------------------------===//
/// Returns vector::ContractionOp operand's index where the result is used.
static std::optional<int>
getVectorContractOpOperandId(vector::ContractionOp contractOp,
                             OpResult result) {
  if (contractOp.getLhs() == result)
    return 0;
  if (contractOp.getRhs() == result)
    return 1;
  if (contractOp.getAcc() == result)
    return 2;
  return std::nullopt;
}

/// Returns vector::ContractionOp operand's index  where the
/// vector::TransferReadOp is consumed either consumed directly or via
/// vector::ExtractStridedSliceOp.
static std::optional<int>
getVectorContractOpOperandIdForVectorReadOp(Operation *op) {
  vector::ContractionOp contractOp;

  // Check if the vector::TransferReadOp is consumed directly by
  // vector::ContractionOp.
  if (op->use_empty())
    return std::nullopt;
  Operation *firstLevelUser = *((op->getUsers()).begin());
  if (!firstLevelUser)
    return std::nullopt;
  if (auto contractOp = dyn_cast<vector::ContractionOp>(firstLevelUser))
    return getVectorContractOpOperandId(contractOp, op->getResult(0));

  // Check if the vector::TransferReadOp is consumed indirectly by
  // vector::ContractionOp. Only check until the second level of use-def chain.
  if (firstLevelUser->use_empty())
    return std::nullopt;
  Operation *secondLevelUser = *((firstLevelUser->getUsers()).begin());
  if (!secondLevelUser)
    return std::nullopt;
  if (auto contractOp = dyn_cast<vector::ContractionOp>(secondLevelUser))
    return getVectorContractOpOperandId(contractOp,
                                        firstLevelUser->getResult(0));
  return std::nullopt;
}

/// Helper function to return native size for MMA.SYNC-based operations.
std::optional<SmallVector<int64_t>> getMmaNativeVectorSize(Operation *op) {
  // Shape of native Tensor Core GPU mma.sync operations.
  int64_t mmaShapeM = 16;
  int64_t mmaShapeN = 8;
  int64_t mmaShapeK;

  // Shape the mma.sync warp-level operation.
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    Type sourceType = contract.getLhsType().getElementType();

    // Set mmaShapeK based on sourceType.
    if (sourceType.isInteger(4))
      mmaShapeK = 64;
    else if (sourceType.isInteger(8))
      mmaShapeK = 32;
    else if (sourceType.isF16() || sourceType.isBF16())
      mmaShapeK = 16;
    else if (sourceType.isF32())
      mmaShapeK = 8;
    else {
      LDBG("unsupported shape for vector.contract: ");
      return std::nullopt;
    }

    // Initialize/set the starting dims of the ranked shape, such as batch,
    // to 1.
    SmallVector<int64_t> mmaShape(contract.getIteratorTypes().size() - 3, 1);
    mmaShape.append({mmaShapeM, mmaShapeN, mmaShapeK});
    LLVM_DEBUG({
      llvm::interleaveComma(mmaShape, DBGS() << "shape for vector.contract: ");
      llvm::dbgs() << "\n";
    });
    return mmaShape;
  }

  // Shape of warp-level vector write operation.
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    if (writeOp.getVectorType().getRank() < 2)
      return std::nullopt;
    SmallVector<int64_t> outputShape(writeOp.getVectorType().getRank() - 2, 1);
    outputShape.append({mmaShapeM, mmaShapeN});
    LLVM_DEBUG({
      llvm::interleaveComma(outputShape,
                            DBGS() << "shape for vector.xfer_write: ");
      llvm::dbgs() << "\n";
    });
    return outputShape;
  }

  // Shape of warp-level vector read (load) operation.
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    auto resultVectorType =
        llvm::cast<VectorType>(readOp.getVector().getType());
    Type resultElementType = resultVectorType.getElementType();

    std::optional<int> operandId =
        getVectorContractOpOperandIdForVectorReadOp(op);
    if (!operandId) {
      LLVM_DEBUG({
        DBGS() << "Failed to get operandId for vector::xfer_read: " << *op
               << "\n";
      });
      return std::nullopt;
    }

    // Loading F16 values from Shared Memory to Registers.
    if (resultElementType.isF16() || resultElementType.isBF16()) {
      // For matrixC.
      if (*operandId == 2) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeN});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }

      // For matrixA and matrixB.
      if (*operandId == 0 || *operandId == 1) {
        // MmaSyncOp input operands: matrixA and matrixB.
        // LDSMx1, x2, x4:
        // - LDSMx1 loads a 1 tile  of 8x8.
        // - LDSMx2 loads a 2 tiles of 8x8.
        // - LDSMx4 loads a 4 tiles of 8x8. (in use)
        // IREE uses the largest tiled load, i.e., LDSMx4.

        // MmaSyncOp source operand: matrixC.
        // matrixC is also read/written in tiled block of 16x16. In the pass
        // OptimizeVectorTransfer, matrixC reads are moved above the mainloop
        // and writes are moved below the mainloop. Thus, mma.sync read/write
        // accumulator inplace.
        SmallVector<int64_t> readShape;
        readShape.append({16, 16});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
    }

    // Loading F32 values from Shared Memory to Registers.
    if (resultElementType.isF32()) {
      // Set mmaShapeK for F32 datatype mma.sync.f32.tf32.m16n8k8.
      mmaShapeK = 8;

      // For matrixC.
      if (*operandId == 2) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeN});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
      // For matrixA.
      if (*operandId == 0) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeK});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
      // For matrixB.
      if (*operandId == 1) {
        // Do not use ldmatrix for matrixB.
        // Transfer read ops may need different shapes based on how they are
        // being used. For simplicity just match the shape used by the extract
        // strided op.
        VectorType sliceType;
        for (Operation *users : op->getUsers()) {
          auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
          if (!extract)
            return std::nullopt;
          auto vecType = llvm::cast<VectorType>(extract.getResult().getType());
          if (sliceType && sliceType != vecType)
            return std::nullopt;
          sliceType = vecType;
        }
        LLVM_DEBUG({
          llvm::interleaveComma(sliceType.getShape(),
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return llvm::to_vector(sliceType.getShape());
      }
    }
  }
  LDBG("unsupported shape for " << op->getName().getStringRef());
  return std::nullopt;
}

bool hasSharedMemoryAddressSpace(MemRefType memrefType) {
  auto addrSpace = llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(
      memrefType.getMemorySpace());
  return addrSpace &&
         addrSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

//===----------------------------------------------------------------------===//
// GPU CodeGen op filter
//===----------------------------------------------------------------------===//

/// Returns true if the index map represents a transpose that benefits from
/// shared mem.
bool sharedMemTransposeFilter(AffineMap indexMap) {
  if (!indexMap.isEmpty() && indexMap.isPermutation()) {
    // Ensure that the fasted moving dimension (the last one) is permuted,
    // Otherwise shared memory promotion will not benefit the operation.
    if (indexMap.getDimPosition(indexMap.getNumDims() - 1) !=
        indexMap.getNumDims() - 1) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// GPU UKernel Utils
//===----------------------------------------------------------------------===//

// TODO: Add more popular kernels into this list and the ukernel cmake.
//       No real technical reason to only allow these aside from compile
//       time and diskspace.
bool hasUkernelSupportedRocmArch(StringRef targetChip) {
  const char *kSupportedTargetChip[] = {"gfx90a", "gfx940", "gfx1030",
                                        "gfx1100"};
  size_t arraySize =
      sizeof(kSupportedTargetChip) / sizeof(kSupportedTargetChip[0]);
  for (int i = 0; i < arraySize; i++) {
    // return true if targetChip is found inside kSupportedTargetChip.
    if (targetChip.compare(kSupportedTargetChip[i]) == 0)
      return true;
  }
  return false;
}

bool hasUkernelSupportedRocmArch(IREE::HAL::ExecutableTargetAttr targetAttr) {
  auto targetArch = getConfigStringAttr(targetAttr, "target_arch");
  if (!targetArch) {
    return false;
  }
  StringRef targetArchStr = targetArch->getValue();
  return hasUkernelSupportedRocmArch(targetArchStr);
}

/// Checks if target GPU has UKernel support.
bool hasUkernelSupportedGpuArch(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isROCMBackend(targetAttr) && hasUkernelSupportedRocmArch(targetAttr)) {
    return true;
  }
  // TODO: Once plumbed, add a CUDA backend and supported cuda arch check.
  return false;
}

//===----------------------------------------------------------------------===//
// GPU Target Information
//===----------------------------------------------------------------------===//

static constexpr char mmaTypeListName[] = "mma_intrinsics";
static FailureOr<ArrayAttr> getSupportedMmaTypes(DictionaryAttr config) {
  if (!config) {
    return failure();
  }
  ArrayAttr types = dyn_cast_or_null<ArrayAttr>(config.get(mmaTypeListName));
  if (!types) {
    return failure();
  }
  return types;
}

FailureOr<ArrayAttr>
getSupportedMmaTypes(mlir::FunctionOpInterface entryPoint) {
  if (auto variantOp =
          entryPoint->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTarget();
    return getSupportedMmaTypes(targetAttr.getConfiguration());
  }
  return failure();
}

} // namespace mlir::iree_compiler
