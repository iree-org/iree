// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements a pass to vectorize scalar interface memrefs into vector
// ones in order to perform vector load/store on these memrefs to achieve better
// memory access patterns.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-vectorize-load-store"

constexpr int kMaxVectorNumBits = 128;
constexpr int kMaxVectorNumElements = 4;

namespace mlir {
namespace iree_compiler {

/// Scans all uses of the given memref `value` to make sure they are ops that we
/// can vectorize, including vector transfer ops and GPU subgroup MMA ops, or
/// other ops that doesn't care. If so, places all vector transfer or GPU
/// subgroup MMA ops in `uses` and returns true.
static bool getUsesIfAllTransferOp(Value value,
                                   SmallVectorImpl<Operation *> &uses) {
  assert(uses.empty() && "expected uses to be empty");
  for (Operation *userOp : value.getUsers()) {
    if (isa<memref::DeallocOp, memref::AssumeAlignmentOp>(userOp)) continue;

    if (!isa<gpu::SubgroupMmaLoadMatrixOp, gpu::SubgroupMmaStoreMatrixOp,
             vector::TransferReadOp, vector::TransferWriteOp>(userOp)) {
      uses.clear();
      LLVM_DEBUG(llvm::dbgs()
                 << "failed: non-transfer-like user: " << *userOp << "\n");
      return false;
    }

    if (auto transferOp = dyn_cast<VectorTransferOpInterface>(userOp)) {
      if (!transferOp.permutation_map().isMinorIdentity()) {
        uses.clear();
        LLVM_DEBUG(llvm::dbgs() << "failed: non-minor-identity transfer user: "
                                << *userOp << "\n");
        return false;
      }
    }

    uses.push_back(userOp);
  }
  return true;
}

/// Returns the bitwidth of a scalar or vector type.
static std::optional<unsigned> getBitWidth(Type type) {
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth();
  }
  if (type.isa<VectorType>()) {
    auto vecType = type.cast<VectorType>();
    auto elementType = vecType.getElementType();
    return elementType.getIntOrFloatBitWidth() * vecType.getNumElements();
  }
  return {};
}

// Calculates the vector bit count we want to use based on the memref uses.
static unsigned calculateMemRefVectorNumBits(
    SmallVectorImpl<Operation *> &uses) {
  unsigned minBits = kMaxVectorNumBits;
  for (Operation *op : uses) {
    if (isa<gpu::SubgroupMmaLoadMatrixOp, gpu::SubgroupMmaStoreMatrixOp>(op)) {
      // We look at this in the next step.
      continue;
    }
    auto transferOp = dyn_cast<VectorTransferOpInterface>(op);
    if (!transferOp) return 0;
    std::optional<unsigned> transferSize =
        getBitWidth(transferOp.getVectorType());
    if (!transferSize) return 0;
    minBits = std::min(minBits, *transferSize);
  }

  for (Operation *op : uses) {
    Value memrefVal;
    int64_t stride;
    if (auto loadOp = dyn_cast<gpu::SubgroupMmaLoadMatrixOp>(op)) {
      memrefVal = loadOp.getSrcMemref();
      stride = loadOp.getLeadDimension().getSExtValue();
    } else if (auto storeOp = dyn_cast<gpu::SubgroupMmaStoreMatrixOp>(op)) {
      memrefVal = storeOp.getDstMemref();
      stride = storeOp.getLeadDimension().getSExtValue();
    }
    if (!memrefVal) continue;

    // GPU subgroup MMA ops do not care about the memref element type. But we
    // still need to make sure we can load/store with good strides.
    // The `leadingDimension` attributes specifies the stride (numer of
    // *elements*) over the memref for the leading dimension.
    auto memrefType = memrefVal.getType().cast<MemRefType>();
    std::optional<unsigned> elementBits =
        getBitWidth(memrefType.getElementType());
    if (!elementBits) return 0;
    int64_t strideBits = stride * *elementBits;
    // Make sure the stride is aligned with the planned vector bitwidth.
    if (strideBits % minBits != 0) return 0;
  }

  return minBits;
}

/// If the memref is vectorizable return the vector bit count we want to use,
/// otherwise return 0. If it returns a value greater than 0 it also returns the
/// memref uses.
static unsigned isMemRefVectorizable(Value value,
                                     SmallVectorImpl<Operation *> &uses) {
  auto memrefType = value.getType().dyn_cast<MemRefType>();

  // Require scalar element type
  if (!memrefType || (!memrefType.getElementType().isa<IntegerType>() &&
                      !memrefType.getElementType().isa<FloatType>())) {
    LLVM_DEBUG(llvm::dbgs() << "failed: not (scalar) memref\n");
    return 0;
  }

  // Require static innermost dimension.
  if (memrefType.getRank() == 0 ||
      ShapedType::isDynamic(memrefType.getShape().back())) {
    LLVM_DEBUG(llvm::dbgs() << "failed: 0 rank or dynamic shape\n");
    return 0;
  }
  const int64_t lastDimSize = memrefType.getShape().back();
  LLVM_DEBUG(llvm::dbgs() << "lastDimSize=" << lastDimSize << "\n");

  // If we have an odd number of elements, it will require padding in the
  // buffer.
  if (lastDimSize % 2 != 0) {
    LLVM_DEBUG(llvm::dbgs() << "failed: innermost dim not divisible by 2\n");
    return 0;
  }

  unsigned elementNumBits = memrefType.getElementTypeBitWidth();
  if (kMaxVectorNumBits % elementNumBits != 0) {
    LLVM_DEBUG(llvm::dbgs() << "failed: element not fitting in vector4\n");
    return 0;
  }

  if (getUsesIfAllTransferOp(value, uses)) {
    unsigned vectorBits = calculateMemRefVectorNumBits(uses);
    if (!vectorBits) return 0;
    unsigned vectorSize = vectorBits / elementNumBits;
    LLVM_DEBUG(llvm::dbgs() << "vectorBits=" << vectorBits << "\n");
    LLVM_DEBUG(llvm::dbgs() << "elementNumBits=" << elementNumBits << "\n");
    // Again make sure we don't have vectors of odd numbers.
    if (vectorSize % 2 != 0) {
      LLVM_DEBUG(llvm::dbgs() << "failed: odd element count after grouping\n");
      return 0;
    }
    if ((lastDimSize * elementNumBits) % vectorBits != 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "failed: innermost dim not divisible by vector size\n");
      return 0;
    }
    return vectorBits;
  }

  return 0;
}

namespace {
/// Analyze memref usages to decide if it should be vectorized. Right now the
/// logic is to vectorize memref only if it is used by vector transfer
/// read/write ops.
class MemRefUsageAnalysis {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemRefUsageAnalysis)

  explicit MemRefUsageAnalysis(mlir::Operation *);

  // Returns true if the memref should be converted to a memref of vectors.
  bool shouldVectorizeMemRef(Value value) const {
    return valueToVectorBitsMap.count(value);
  }

  // Return the size of the vector we want to use for memref vectorization.
  unsigned getMemRefVectorSizeInBits(Value value) const {
    return valueToVectorBitsMap.find(value)->second;
  }
  // Returns true if the transfer operation needs to be updated during memref
  // vectorization.
  bool shouldConvertTransfer(Operation *op) const {
    return transferOps.count(op);
  }

 private:
  void analyzeMemRefValue(Value value);

  // The mapping from a MemRef value to the number of bits of the vector this
  // MemRef value should be vectorized into.
  llvm::DenseMap<Value, unsigned> valueToVectorBitsMap;
  // A list of transfer ops that should be adjusted for memref vectorization.
  llvm::DenseSet<Operation *> transferOps;
};

MemRefUsageAnalysis::MemRefUsageAnalysis(mlir::Operation *op) {
  op->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<func::FuncOp>([this](func::FuncOp funcOp) {
          for (Value arg : funcOp.getArguments()) {
            analyzeMemRefValue(arg);
          }
        })
        .Case<memref::AllocOp, IREE::HAL::InterfaceBindingSubspanOp>(
            [this](auto op) { analyzeMemRefValue(op); });
  });
}

void MemRefUsageAnalysis::analyzeMemRefValue(Value value) {
  SmallVector<Operation *, 4> vectorUses;
  LLVM_DEBUG(llvm::dbgs() << "analyzing value: " << value << "\n");
  if (unsigned vectorSize = isMemRefVectorizable(value, vectorUses)) {
    valueToVectorBitsMap.insert(std::make_pair(value, vectorSize));
    transferOps.insert(vectorUses.begin(), vectorUses.end());
  }
}

template <typename OpTy>
class MemRefConversionPattern : public OpConversionPattern<OpTy> {
 public:
  MemRefConversionPattern<OpTy>(MLIRContext *context,
                                const MemRefUsageAnalysis &memrefUsageAnalysis)
      : OpConversionPattern<OpTy>::OpConversionPattern(context),
        memrefUsageAnalysis(memrefUsageAnalysis) {}

 protected:
  std::optional<MemRefType> getVectorizedMemRefType(
      ConversionPatternRewriter &rewriter, Value memRefValue) const;

  /// Adjusts indices for vector transfer / GPU MMA load/store ops to index into
  /// vector memref.
  FailureOr<SmallVector<Value>> adjustIndices(
      MemRefType scalarMemrefType, MemRefType vectorMemrefType,
      ValueRange indices, ConversionPatternRewriter &rewriter,
      Location loc) const;

  const MemRefUsageAnalysis &memrefUsageAnalysis;
};

class ProcessFunctionArgument final
    : public MemRefConversionPattern<func::FuncOp> {
 public:
  using MemRefConversionPattern::MemRefConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp funcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

class ProcessTransferRead final
    : public MemRefConversionPattern<vector::TransferReadOp> {
 public:
  using MemRefConversionPattern::MemRefConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp read, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!memrefUsageAnalysis.shouldConvertTransfer(read)) {
      return rewriter.notifyMatchFailure(
          read, "cannot be vectorized per memref usage analysis");
    }

    assert(read.getPermutationMap().isMinorIdentity());

    Location loc = read.getLoc();

    auto scalarMemrefType = read.getSource().getType().dyn_cast<MemRefType>();
    auto vectorMemrefType =
        adaptor.getSource().getType().dyn_cast<MemRefType>();
    auto readVectorType = read.getVectorType();
    if (!scalarMemrefType || !vectorMemrefType) return failure();

    std::optional<unsigned> vectorMemrefElemSize =
        getBitWidth(vectorMemrefType.getElementType());
    std::optional<unsigned> readVecSize = getBitWidth(readVectorType);

    auto indices = adjustIndices(scalarMemrefType, vectorMemrefType,
                                 adaptor.getIndices(), rewriter, loc);
    if (failed(indices)) return failure();

    // If the transfer_read can be replaced by a load after vectorization use
    // LoadOp and cast back to the original type.
    if (*vectorMemrefElemSize == *readVecSize) {
      Type elemType = vectorMemrefType.getElementType();
      Value newLoad = rewriter.create<memref::LoadOp>(
          loc, elemType, adaptor.getSource(), indices.value());
      Type serializedVecType =
          VectorType::get(read.getVectorType().getNumElements(),
                          read.getVectorType().getElementType());
      newLoad =
          rewriter.create<vector::BitCastOp>(loc, serializedVecType, newLoad);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
          read, read.getVectorType(), newLoad);
    } else {
      rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
          read, read.getVectorType(), adaptor.getSource(), indices.value());
    }
    return success();
  }
};

class ProcessTransferWrite final
    : public MemRefConversionPattern<vector::TransferWriteOp> {
 public:
  using MemRefConversionPattern::MemRefConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp write, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!memrefUsageAnalysis.shouldConvertTransfer(write)) {
      return rewriter.notifyMatchFailure(
          write, "cannot be vectorized per memref usage analysis");
    }

    assert(write.getPermutationMap().isMinorIdentity());

    Location loc = write.getLoc();

    auto scalarMemrefType = write.getSource().getType().dyn_cast<MemRefType>();
    auto vectorMemrefType =
        adaptor.getSource().getType().dyn_cast<MemRefType>();
    auto writeVectorType = write.getVectorType();
    if (!scalarMemrefType || !vectorMemrefType) return failure();

    std::optional<unsigned> vectorMemrefElemSize =
        getBitWidth(vectorMemrefType.getElementType());
    std::optional<unsigned> writeVecSize = getBitWidth(writeVectorType);

    auto indices = adjustIndices(scalarMemrefType, vectorMemrefType,
                                 adaptor.getIndices(), rewriter, loc);
    if (failed(indices)) return failure();

    // If the transfer_write can be replaced by a store after vectorization cast
    // the original value and use StoreOp.
    if (*vectorMemrefElemSize == *writeVecSize) {
      Type serializedVecType = VectorType::get(
          writeVectorType.getNumElements(), writeVectorType.getElementType());
      Value data = rewriter.create<vector::ShapeCastOp>(loc, serializedVecType,
                                                        adaptor.getVector());
      data = rewriter.create<vector::BitCastOp>(
          loc, vectorMemrefType.getElementType(), data);
      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          write, data, adaptor.getSource(), indices.value());
    } else {
      rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
          write, adaptor.getVector(), adaptor.getSource(), indices.value());
    }
    return success();
  }
};

/// Decide the new memref of vector type we want to use after vectorization
/// based on the original type and the vectorization size we want. Since Vulkan
/// only supports vector up to 4 elements we may re-interpret the memref using a
/// larger type. For example:
/// * memref<1024xf16> vectorized with a size of 64bits will return
/// memref<256xvec<4xf16>>
/// * memref<1024xf16> vectorized with a size of 128bits will return
/// memref<128xvec<4xf32>>
template <typename OpTy>
std::optional<MemRefType>
MemRefConversionPattern<OpTy>::getVectorizedMemRefType(
    ConversionPatternRewriter &rewriter, Value memRefValue) const {
  MemRefType type = memRefValue.getType().cast<MemRefType>();
  unsigned vectorNumBits =
      memrefUsageAnalysis.getMemRefVectorSizeInBits(memRefValue);

  Type scalarType = type.getElementType();
  unsigned scalarNumBits = type.getElementTypeBitWidth();
  unsigned vectorNumElements = vectorNumBits / scalarNumBits;
  // If the vector we need to generate is bigger than the the max vector size
  // allowed for loads use a larger element type.
  if (vectorNumElements > kMaxVectorNumElements) {
    scalarType = scalarType.isa<IntegerType>()
                     ? rewriter.getI32Type().cast<Type>()
                     : rewriter.getF32Type().cast<Type>();
    scalarNumBits = scalarType.getIntOrFloatBitWidth();
    vectorNumElements = vectorNumBits / scalarNumBits;
  }

  Type vectorType = VectorType::get(vectorNumElements, scalarType);
  auto newShape = llvm::to_vector<2>(type.getShape());
  unsigned ratio = vectorNumBits / type.getElementTypeBitWidth();
  if (newShape.back() % ratio != 0) return {};
  newShape.back() = newShape.back() / ratio;

  MemRefLayoutAttrInterface layout = {};
  if (auto stridedLayout = type.getLayout().dyn_cast<StridedLayoutAttr>()) {
    auto offset = stridedLayout.getOffset();
    if (offset != ShapedType::kDynamic) {
      offset = offset / ratio;
    }

    auto strides = llvm::to_vector(stridedLayout.getStrides());
    for (auto [index, stride] : llvm::enumerate(llvm::drop_end(strides))) {
      if (index == strides.size() - 1 || stride == ShapedType::kDynamic) {
        continue;
      }
      strides[index] = stride / ratio;
    }
    layout = StridedLayoutAttr::get(rewriter.getContext(), offset, strides);
  }

  return MemRefType::get(newShape, vectorType, layout, type.getMemorySpace());
}

template <typename OpTy>
FailureOr<SmallVector<Value>> MemRefConversionPattern<OpTy>::adjustIndices(
    MemRefType scalarMemrefType, MemRefType vectorMemrefType,
    ValueRange indices, ConversionPatternRewriter &rewriter,
    Location loc) const {
  std::optional<unsigned> vectorMemrefElemSize =
      getBitWidth(vectorMemrefType.getElementType());
  std::optional<unsigned> scalarMemrefElemSize =
      getBitWidth(scalarMemrefType.getElementType());
  if (!vectorMemrefElemSize || !scalarMemrefElemSize) return failure();

  MLIRContext *context = rewriter.getContext();
  AffineExpr sym0, sym1;
  bindSymbols(context, sym0, sym1);
  auto divMap = AffineMap::get(0, 2, {sym0.floorDiv(sym1)}, context);

  unsigned ratio = *vectorMemrefElemSize / *scalarMemrefElemSize;
  Value valueRatio = rewriter.create<arith::ConstantIndexOp>(loc, ratio);
  auto newIndices = llvm::to_vector(indices);
  newIndices.back() = rewriter.create<affine::AffineApplyOp>(
      loc, divMap, ValueRange{indices.back(), valueRatio});
  return newIndices;
}

class ProcessAlloc final : public MemRefConversionPattern<memref::AllocOp> {
 public:
  using MemRefConversionPattern::MemRefConversionPattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp alloc, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto memrefType = getVectorizedMemRefType(rewriter, alloc.getResult());
    if (!memrefType) return failure();
    rewriter.replaceOpWithNewOp<memref::AllocOp>(alloc, *memrefType,
                                                 alloc.getDynamicSizes());
    return success();
  }
};

class ProcessInterfaceBindingSubspan final
    : public MemRefConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
 public:
  using MemRefConversionPattern::MemRefConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp subspanOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto memrefType = subspanOp.getType().dyn_cast<MemRefType>();
    if (!memrefType) return failure();

    // This should be guaranteed by the analysis step. But just double check.
    assert(memrefType.getRank() > 0 &&
           !ShapedType::isDynamic(memrefType.getShape().back()));

    auto vecMemRef = getVectorizedMemRefType(rewriter, subspanOp.getResult());
    if (!vecMemRef) {
      return rewriter.notifyMatchFailure(subspanOp,
                                         "cannot get vectorized memref type");
    }
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, *vecMemRef, subspanOp.getSet(), subspanOp.getBinding(),
        subspanOp.getDescriptorType(), subspanOp.getByteOffset(),
        subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());
    return success();
  }
};

struct ProcessSubgroupMMALoad final
    : public MemRefConversionPattern<gpu::SubgroupMmaLoadMatrixOp> {
  using MemRefConversionPattern::MemRefConversionPattern;

  LogicalResult matchAndRewrite(
      gpu::SubgroupMmaLoadMatrixOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto scalarMemrefType =
        loadOp.getSrcMemref().getType().dyn_cast<MemRefType>();
    auto vectorMemrefType =
        adaptor.getSrcMemref().getType().dyn_cast<MemRefType>();

    Location loc = loadOp.getLoc();
    auto indices = adjustIndices(scalarMemrefType, vectorMemrefType,
                                 adaptor.getIndices(), rewriter, loc);
    if (failed(indices)) return failure();

    // Compute how many bits the mma op stride corresponds to for the scalar
    // memref, and rescale it to vector memref.
    int64_t stride = loadOp.getLeadDimension().getSExtValue();
    auto scalarBits = getBitWidth(scalarMemrefType.getElementType());
    auto vectorBits = getBitWidth(vectorMemrefType.getElementType());
    int64_t strideBits = stride * *scalarBits;
    auto newLeadDimSize = rewriter.getIntegerAttr(
        loadOp.getLeadDimensionAttr().getType(), strideBits / *vectorBits);

    rewriter.replaceOpWithNewOp<gpu::SubgroupMmaLoadMatrixOp>(
        loadOp, loadOp.getType(), adaptor.getSrcMemref(), indices.value(),
        newLeadDimSize, loadOp.getTransposeAttr());
    return success();
  }
};

struct ProcessSubgroupMMAStore final
    : public MemRefConversionPattern<gpu::SubgroupMmaStoreMatrixOp> {
  using MemRefConversionPattern::MemRefConversionPattern;

  LogicalResult matchAndRewrite(
      gpu::SubgroupMmaStoreMatrixOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto scalarMemrefType =
        storeOp.getDstMemref().getType().dyn_cast<MemRefType>();
    auto vectorMemrefType =
        adaptor.getDstMemref().getType().dyn_cast<MemRefType>();

    Location loc = storeOp.getLoc();
    auto indices = adjustIndices(scalarMemrefType, vectorMemrefType,
                                 adaptor.getIndices(), rewriter, loc);
    if (failed(indices)) return failure();

    // Compute how many bits the mma op stride corresponds to for the scalar
    // memref, and rescale it to vector memref.
    int64_t stride = storeOp.getLeadDimension().getSExtValue();
    auto scalarBits = getBitWidth(scalarMemrefType.getElementType());
    auto vectorBits = getBitWidth(vectorMemrefType.getElementType());
    int64_t strideBits = stride * *scalarBits;
    auto newLeadDimSize = rewriter.getIntegerAttr(
        storeOp.getLeadDimensionAttr().getType(), strideBits / *vectorBits);

    rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
        storeOp, adaptor.getSrc(), adaptor.getDstMemref(), indices.value(),
        newLeadDimSize, storeOp.getTransposeAttr());
    return success();
  }
};

template <typename OpT>
class PassThroughConversion : public OpConversionPattern<OpT> {
 public:
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpT op, typename OpT::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

/// Scalarizes remaining vector transfer that couldn't be converted to
/// vevtor load operations.

/// This is very specific to SPIR-V as pointer cannot be casted to vector type
/// if any of the memory access is not vector.
struct ScalarizeVectorTransferRead final
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = readOp.getType();
    auto map = readOp.getPermutationMap();
    if (vectorType.getRank() > 1 || !map.isProjectedPermutation())
      return failure();

    Location loc = readOp.getLoc();
    if (vectorType.getRank() == 0) {
      Value scalar = rewriter.create<memref::LoadOp>(loc, readOp.getSource(),
                                                     readOp.getIndices());
      rewriter.replaceOpWithNewOp<vector::SplatOp>(readOp, vectorType, scalar);
      return success();
    }

    MLIRContext *context = rewriter.getContext();
    AffineExpr sym0, sym1;
    bindSymbols(context, sym0, sym1);
    auto addMap = AffineMap::get(0, 2, {sym0 + sym1}, context);

    // The result vector is 1-D and we have a projected permutation.
    unsigned dimPos = map.getDimPosition(0);

    auto indices = llvm::to_vector<4>(readOp.getIndices());
    Value oldIndex = indices[dimPos];

    Value newVector = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));
    for (int i = 0; i < vectorType.getDimSize(0); ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      indices[dimPos] = rewriter.create<affine::AffineApplyOp>(
          loc, addMap, ValueRange{oldIndex, iVal});
      Value scalar =
          rewriter.create<memref::LoadOp>(loc, readOp.getSource(), indices);
      newVector = rewriter.create<vector::InsertOp>(loc, scalar, newVector, i);
    }
    rewriter.replaceOp(readOp, newVector);
    return success();
  }
};

struct ScalarizeVectorLoad final : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = loadOp.getType();
    if (vectorType.getRank() > 1) return failure();

    Location loc = loadOp.getLoc();
    if (vectorType.getRank() == 0) {
      Value scalar = rewriter.create<memref::LoadOp>(loc, loadOp.getBase(),
                                                     loadOp.getIndices());
      rewriter.replaceOpWithNewOp<vector::SplatOp>(loadOp, vectorType, scalar);
      return success();
    }

    MLIRContext *context = rewriter.getContext();
    AffineExpr sym0, sym1;
    bindSymbols(context, sym0, sym1);
    auto addMap = AffineMap::get(0, 2, {sym0 + sym1}, context);

    // The result vector is 1-D so we just unroll the load along the last index.
    unsigned dimPos = loadOp.getBase().getType().getRank() - 1;

    auto indices = llvm::to_vector<4>(loadOp.getIndices());
    Value oldIndex = indices[dimPos];

    Value newVector = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));
    for (int i = 0; i < vectorType.getDimSize(0); ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      indices[dimPos] = rewriter.create<affine::AffineApplyOp>(
          loc, addMap, ValueRange{oldIndex, iVal});
      Value scalar =
          rewriter.create<memref::LoadOp>(loc, loadOp.getBase(), indices);
      newVector = rewriter.create<vector::InsertOp>(loc, scalar, newVector, i);
    }
    rewriter.replaceOp(loadOp, newVector);
    return success();
  }
};

struct ScalarizeVectorTransferWrite final
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = writeOp.getVectorType();
    auto map = writeOp.getPermutationMap();
    if (vectorType.getRank() > 1 || !map.isProjectedPermutation())
      return failure();

    Location loc = writeOp.getLoc();
    if (vectorType.getRank() == 0) {
      Value scalar =
          rewriter.create<vector::ExtractElementOp>(loc, writeOp.getVector());
      rewriter.create<memref::StoreOp>(loc, scalar, writeOp.getSource(),
                                       writeOp.getIndices());
      rewriter.eraseOp(writeOp);
      return success();
    }

    MLIRContext *context = rewriter.getContext();
    AffineExpr sym0, sym1;
    bindSymbols(context, sym0, sym1);
    auto addMap = AffineMap::get(0, 2, {sym0 + sym1}, context);

    // The result vector is 1-D and we have a projected permutation.
    unsigned dimPos = map.getDimPosition(0);

    auto indices = llvm::to_vector<4>(writeOp.getIndices());
    Value oldIndex = indices[dimPos];
    for (int i = 0; i < vectorType.getDimSize(0); ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      indices[dimPos] = rewriter.create<affine::AffineApplyOp>(
          loc, addMap, ValueRange{oldIndex, iVal});
      Value scalar =
          rewriter.create<vector::ExtractOp>(loc, writeOp.getVector(), i);
      rewriter.create<memref::StoreOp>(loc, scalar, writeOp.getSource(),
                                       indices);
    }
    rewriter.eraseOp(writeOp);
    return success();
  }
};

class SPIRVVectorizeLoadStorePass final
    : public SPIRVVectorizeLoadStoreBase<SPIRVVectorizeLoadStorePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override;

 private:
  MemRefUsageAnalysis *memrefUsageAnalysis = nullptr;
};
}  // namespace

LogicalResult ProcessFunctionArgument::matchAndRewrite(
    func::FuncOp funcOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  TypeConverter::SignatureConversion signatureConverter(
      funcOp.getFunctionType().getNumInputs());
  for (const auto &[index, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (memrefUsageAnalysis.shouldVectorizeMemRef(arg)) {
      if (auto memrefType = getVectorizedMemRefType(rewriter, arg)) {
        signatureConverter.addInputs(index, *memrefType);
        continue;
      }
    }
    signatureConverter.addInputs(index, arg.getType());
  }
  // Creates a new function with the update signature.
  rewriter.applySignatureConversion(&funcOp.getFunctionBody(),
                                    signatureConverter);

  // Creates a new function with the update signature.
  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.setType(rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), std::nullopt));
  });
  return success();
}

void SPIRVVectorizeLoadStorePass::runOnOperation() {
  // Uses the signature conversion methodology of the dialect conversion
  // framework to implement the conversion.
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  memrefUsageAnalysis = &getAnalysis<MemRefUsageAnalysis>();

  RewritePatternSet conversionPatterns(context);
  conversionPatterns
      .add<ProcessFunctionArgument, ProcessTransferRead, ProcessTransferWrite,
           ProcessSubgroupMMALoad, ProcessSubgroupMMAStore, ProcessAlloc,
           ProcessInterfaceBindingSubspan>(context, *memrefUsageAnalysis);
  conversionPatterns.add<PassThroughConversion<memref::DeallocOp>,
                         PassThroughConversion<memref::AssumeAlignmentOp>>(
      context);

  ConversionTarget target(*context);
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return llvm::all_of(op.getArguments(), [&](Value arg) {
      return !memrefUsageAnalysis->shouldVectorizeMemRef(arg);
    });
  });
  target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp alloc) {
    return !memrefUsageAnalysis->shouldVectorizeMemRef(alloc);
  });
  target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
    return !memrefUsageAnalysis->shouldVectorizeMemRef(op.getMemref());
  });
  target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
      [&](IREE::HAL::InterfaceBindingSubspanOp bindingOp) {
        return !memrefUsageAnalysis->shouldVectorizeMemRef(bindingOp);
      });
  target.addDynamicallyLegalOp<memref::AssumeAlignmentOp>(
      [&](memref::AssumeAlignmentOp op) {
        return !memrefUsageAnalysis->shouldVectorizeMemRef(op.getMemref());
      });
  target.addDynamicallyLegalOp<gpu::SubgroupMmaLoadMatrixOp,
                               gpu::SubgroupMmaStoreMatrixOp,
                               vector::TransferReadOp, vector::TransferWriteOp>(
      [&](auto op) { return !memrefUsageAnalysis->shouldConvertTransfer(op); });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) { return true; });

  if (failed(applyPartialConversion(module, target,
                                    std::move(conversionPatterns)))) {
    return signalPassFailure();
  }

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    RewritePatternSet rewritingPatterns(context);
    rewritingPatterns.add<ScalarizeVectorTransferRead, ScalarizeVectorLoad,
                          ScalarizeVectorTransferWrite>(context);

    if (failed(
            applyPatternsAndFoldGreedily(func, std::move(rewritingPatterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createSPIRVVectorizeLoadStore() {
  return std::make_unique<SPIRVVectorizeLoadStorePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
