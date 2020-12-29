// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===----------------------------------------------------------------------===//
//
// Pass to convert memref into memref of vector.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

constexpr int kMaxVectorizationSizeInBits = 128;
constexpr int kMaxVectorNumElements = 4;

namespace mlir {
namespace iree_compiler {

/// Returns true if all uses are transfer read/write operations. If it returns
/// true also return the uses of memref.
static bool getUsesIfAllTransferOp(Value v,
                                   SmallVectorImpl<Operation *> &uses) {
  assert(uses.empty() && "expected uses to be empty");
  for (Operation *userOp : v.getUsers()) {
    if (isa<DeallocOp>(userOp)) continue;
    // Only vectorize memref used by vector transfer ops.
    if (!isa<vector::TransferReadOp, vector::TransferWriteOp>(userOp)) {
      uses.clear();
      return false;
    }
    uses.push_back(userOp);
  }
  return true;
}


/// Returns the bitwidth of a scalar or vector type.
static Optional<unsigned> getBitWidth(Type type) {
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth();
  } else if (type.isa<VectorType>()) {
    auto vecType = type.cast<VectorType>();
    auto elementType = vecType.getElementType();
    return elementType.getIntOrFloatBitWidth() * vecType.getNumElements();
  }
  return {};
}

// Calculate the vector size we want to use based on the memref uses.
static unsigned calculateMemrefVecSize(SmallVectorImpl<Operation *> &uses) {
  unsigned minSize = kMaxVectorizationSizeInBits;
  for (Operation *op : uses) {
    auto transferOp = dyn_cast<VectorTransferOpInterface>(op);
    if (!transferOp) return 0;
    Optional<unsigned> transferSize = getBitWidth(transferOp.getVectorType());
    if (!transferSize) return 0;
    minSize = std::min(minSize, *transferSize);
  }
  return minSize;
}

/// If the memref is vectorizable return the vector size we want to use,
/// otherwise return 0. If it returns a value greater than 0 it also returns the
/// memref uses.
static unsigned isMemRefAndVectorizable(Value v,
                                        SmallVectorImpl<Operation *> &uses) {
  auto memrefType = v.getType().dyn_cast<MemRefType>();
  // To be able to vectorize the memref it needs to be a scalar memref with a
  // static most inner dimension aligned on the vectorization size.
  if (memrefType && !memrefType.getElementType().isa<VectorType>() &&
      (kMaxVectorizationSizeInBits % memrefType.getElementTypeBitWidth() ==
       0) &&
      memrefType.getRank() > 0 &&
      !ShapedType::isDynamic(memrefType.getShape().back()) &&
      getUsesIfAllTransferOp(v, uses)) {
    return calculateMemrefVecSize(uses);
  }
  return 0;
}

namespace {
/// Analyze memref usages to decide if it should be vectorized. Right now the
/// logic is to vectorize memref only if it is used by
/// vectortransfer_read/vectortransfer_write operations.
class MemRefUsageAnalysis {
 public:
  explicit MemRefUsageAnalysis(mlir::Operation *);

  // Returns true if the memref should be converted to a vector of memref.
  bool vectorizeMemRef(Value v) const { return vectorization_size.count(v); }

  // Return the size of the vector we want to use for memref vectorization.
  unsigned getMemRefVectorSizeInBits(Value v) const {
    return vectorization_size.find(v)->second;
  }
  // Returns true if the transfer operation needs to be updated during memref
  // vectorization.
  bool transferConvert(Operation *op) const { return transferOps.count(op); }

 private:
  void analyzeFunc(FuncOp funcOp);
  void analyzeAlloc(AllocOp allocOp);
  void analyzePlaceholder(IREE::PlaceholderOp placeholderOp);
  llvm::DenseMap<Value, unsigned> vectorization_size;
  llvm::DenseSet<Operation *> transferOps;
};

MemRefUsageAnalysis::MemRefUsageAnalysis(mlir::Operation *op) {
  op->walk([&](Operation *op) {
    if (auto func = dyn_cast<FuncOp>(op)) analyzeFunc(func);
    if (auto alloc = dyn_cast<AllocOp>(op)) analyzeAlloc(alloc);
    if (auto placeholder = dyn_cast<IREE::PlaceholderOp>(op))
      analyzePlaceholder(placeholder);
  });
}

void MemRefUsageAnalysis::analyzeFunc(FuncOp funcOp) {
  for (Value arg : funcOp.getArguments()) {
    SmallVector<Operation *, 4> vectorUses;
    if (unsigned vectorSize = isMemRefAndVectorizable(arg, vectorUses)) {
      vectorization_size.insert(std::make_pair(arg, vectorSize));
      transferOps.insert(vectorUses.begin(), vectorUses.end());
    }
  }
}

void MemRefUsageAnalysis::analyzePlaceholder(
    IREE::PlaceholderOp placeholderOp) {
  SmallVector<Operation *, 4> vectorUses;
  if (unsigned vectorSize =
          isMemRefAndVectorizable(placeholderOp, vectorUses)) {
    vectorization_size.insert(std::make_pair(placeholderOp, vectorSize));
    transferOps.insert(vectorUses.begin(), vectorUses.end());
  }
}

void MemRefUsageAnalysis::analyzeAlloc(AllocOp allocOp) {
  SmallVector<Operation *, 4> vectorUses;
  if (unsigned vectorSize = isMemRefAndVectorizable(allocOp, vectorUses)) {
    vectorization_size.insert(std::make_pair(allocOp, vectorSize));
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
  Optional<MemRefType> getVectorizedMemRefType(
      ConversionPatternRewriter &rewriter, Value memRefValue) const;
  const MemRefUsageAnalysis &memrefUsageAnalysis;
};

class ProcessFuncArg final : public MemRefConversionPattern<FuncOp> {
 public:
  using MemRefConversionPattern<FuncOp>::MemRefConversionPattern;
  LogicalResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

class ProcessTransferRead final
    : public MemRefConversionPattern<vector::TransferReadOp> {
 public:
  using MemRefConversionPattern<
      vector::TransferReadOp>::MemRefConversionPattern;
  LogicalResult matchAndRewrite(
      vector::TransferReadOp read, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!memrefUsageAnalysis.transferConvert(read)) return failure();
    vector::TransferReadOp::Adaptor adaptor(operands);
    Value memref = adaptor.source();
    auto memrefType = memref.getType().dyn_cast<MemRefType>();
    if (!memrefType) return failure();
    Location loc = read.getLoc();
    Optional<unsigned> vecMemrefElemSize =
        getBitWidth(memrefType.getElementType());
    Optional<unsigned> readElemSize = getBitWidth(memrefType.getElementType());
    Optional<unsigned> readVecSize = getBitWidth(read.getVectorType());
    if (!vecMemrefElemSize || !readElemSize || !readVecSize) return failure();
    unsigned ratio = *vecMemrefElemSize / *readElemSize;
    SmallVector<Value, 4> indices(adaptor.indices().begin(),
                                  adaptor.indices().end());
    indices.back() = rewriter.create<SignedDivIOp>(
        loc, indices.back(), rewriter.create<ConstantIndexOp>(loc, ratio));
    // If the transfer_read can be replaced by a load after vectorization use
    // LoadOp and cast back to the original type.
    if (*vecMemrefElemSize == *readVecSize) {
      Type elemType = memrefType.getElementType();
      Value newLoad = rewriter.create<LoadOp>(loc, elemType, memref, indices);
      Type serializedVecType =
          VectorType::get(read.getVectorType().getNumElements(),
                          read.getVectorType().getElementType());
      newLoad =
          rewriter.create<vector::BitCastOp>(loc, serializedVecType, newLoad);
      newLoad = rewriter.create<vector::ShapeCastOp>(loc, read.getVectorType(),
                                                     newLoad);
      rewriter.replaceOp(read, newLoad);
    } else {
      Value newRead = rewriter.create<vector::TransferReadOp>(
          loc, read.getVectorType(), memref, indices);
      rewriter.replaceOp(read, newRead);
    }
    return success();
  }
};

class ProcessTransferWrite final
    : public MemRefConversionPattern<vector::TransferWriteOp> {
 public:
  using MemRefConversionPattern<
      vector::TransferWriteOp>::MemRefConversionPattern;
  LogicalResult matchAndRewrite(
      vector::TransferWriteOp write, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!memrefUsageAnalysis.transferConvert(write)) return failure();
    vector::TransferWriteOp::Adaptor adaptor(operands);
    Value memref = adaptor.source();
    auto memrefType = memref.getType().dyn_cast<MemRefType>();
    if (!memrefType) return failure();
    Location loc = write.getLoc();
    Optional<unsigned> vecMemrefElemSize =
        getBitWidth(memrefType.getElementType());
    Optional<unsigned> writeElemSize = getBitWidth(memrefType.getElementType());
    Optional<unsigned> writeVecSize = getBitWidth(write.getVectorType());
    if (!vecMemrefElemSize || !writeElemSize || !writeVecSize) return failure();
    unsigned ratio = *vecMemrefElemSize / *writeElemSize;
    SmallVector<Value, 4> indices(adaptor.indices());
    indices.back() = rewriter.create<SignedDivIOp>(
        loc, indices.back(), rewriter.create<ConstantIndexOp>(loc, ratio));
    // If the transfer_write can be replaced by a store after vectorization cast
    // the original value and use StoreOp.
    if (*vecMemrefElemSize == *writeVecSize) {
      Type serializedVecType =
          VectorType::get(write.getVectorType().getNumElements(),
                          write.getVectorType().getElementType());
      Value data = rewriter.create<vector::ShapeCastOp>(loc, serializedVecType,
                                                        adaptor.vector());
      data = rewriter.create<vector::BitCastOp>(
          loc, memrefType.getElementType(), data);
      rewriter.create<StoreOp>(loc, data, memref, indices);
    } else {
      rewriter.create<vector::TransferWriteOp>(loc, adaptor.vector(), memref,
                                               indices);
    }
    rewriter.eraseOp(write);
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
Optional<MemRefType> MemRefConversionPattern<OpTy>::getVectorizedMemRefType(
    ConversionPatternRewriter &rewriter, Value memRefValue) const {
  unsigned vecSizeInBits =
      memrefUsageAnalysis.getMemRefVectorSizeInBits(memRefValue);
  MemRefType type = memRefValue.getType().cast<MemRefType>();
  unsigned elemSize = type.getElementTypeBitWidth();
  unsigned numElements = vecSizeInBits / elemSize;
  Type elemType = type.getElementType();
  // If the vector we need to generate is bigger than the the max vector size
  // allowed for loads use a larger element type.
  if (numElements > kMaxVectorNumElements) {
    elemType = elemType.isa<IntegerType>() ? rewriter.getI32Type().cast<Type>()
                                           : rewriter.getF32Type().cast<Type>();
    elemSize = elemType.getIntOrFloatBitWidth();
    numElements = vecSizeInBits / elemSize;
  }
  Type vecType = VectorType::get(numElements, elemType);
  SmallVector<int64_t, 2> newShape(type.getShape().begin(),
                                   type.getShape().end());
  unsigned ratio = vecSizeInBits / type.getElementTypeBitWidth();
  if (newShape.back() % ratio != 0) return {};
  newShape.back() = newShape.back() / ratio;
  return MemRefType::get(newShape, vecType, {}, type.getMemorySpace());
}

class ProcessAlloc final : public MemRefConversionPattern<AllocOp> {
 public:
  using MemRefConversionPattern<AllocOp>::MemRefConversionPattern;
  LogicalResult matchAndRewrite(
      AllocOp alloc, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto memrefType = getVectorizedMemRefType(rewriter, alloc.getResult());
    if (!memrefType) return failure();
    Value newAlloc = rewriter.create<AllocOp>(alloc.getLoc(), *memrefType,
                                              alloc.dynamicSizes());
    rewriter.replaceOp(alloc, newAlloc);
    return success();
  }
};

class ProcessIreeBinding final
    : public MemRefConversionPattern<IREE::PlaceholderOp> {
 public:
  using MemRefConversionPattern<IREE::PlaceholderOp>::MemRefConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::PlaceholderOp placeholder, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto memrefType = placeholder.getType().dyn_cast<MemRefType>();
    if (!memrefType) return failure();
    auto vecMemRef = getVectorizedMemRefType(rewriter, placeholder.getResult());
    if (!vecMemRef) return failure();
    ValueRange dummyOperands;
    Value newPlaceholder = rewriter.create<IREE::PlaceholderOp>(
        placeholder.getLoc(), *vecMemRef, dummyOperands,
        placeholder.getAttrs());
    rewriter.replaceOp(placeholder, newPlaceholder);
    return success();
  }
};

class VectorizeMemRefPass final
    : public PassWrapper<VectorizeMemRefPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;

 private:
  MemRefUsageAnalysis *memrefUsageAnalysis = nullptr;
};
}  // namespace

LogicalResult ProcessFuncArg::matchAndRewrite(
    FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  TypeConverter::SignatureConversion signatureConverter(
      funcOp.getType().getNumInputs());
  TypeConverter typeConverter;
  for (const auto &arg : llvm::enumerate(funcOp.getArguments())) {
    if (memrefUsageAnalysis.vectorizeMemRef(arg.value())) {
      if (auto memrefType = getVectorizedMemRefType(rewriter, arg.value())) {
        signatureConverter.addInputs(arg.index(), *memrefType);
        continue;
      }
    }
    signatureConverter.addInputs(arg.index(), arg.value().getType());
  }
  // Creates a new function with the update signature.
  if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), typeConverter,
                                         &signatureConverter)))
    return failure();

  // Creates a new function with the update signature.
  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.setType(rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), llvm::None));
  });
  return success();
}

void VectorizeMemRefPass::runOnOperation() {
  // Uses the signature conversion methodology of the dialect conversion
  // framework to implement the conversion.
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();
  memrefUsageAnalysis = &getAnalysis<MemRefUsageAnalysis>();

  OwningRewritePatternList patterns;
  patterns.insert<ProcessFuncArg, ProcessTransferRead, ProcessTransferWrite,
                  ProcessAlloc, ProcessIreeBinding>(context,
                                                    *memrefUsageAnalysis);

  ConversionTarget target(*context);
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return llvm::all_of(op.getArguments(), [&](Value arg) {
      return !memrefUsageAnalysis->vectorizeMemRef(arg);
    });
  });
  target.addDynamicallyLegalOp<AllocOp>([&](AllocOp alloc) {
    return !memrefUsageAnalysis->vectorizeMemRef(alloc);
  });
  target.addDynamicallyLegalOp<IREE::PlaceholderOp>(
      [&](IREE::PlaceholderOp placeholder) {
        return !memrefUsageAnalysis->vectorizeMemRef(placeholder);
      });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    if (isa<vector::TransferWriteOp, vector::TransferReadOp>(op))
      return !memrefUsageAnalysis->transferConvert(op);
    return true;
  });
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createVectorizeMemref() {
  return std::make_unique<VectorizeMemRefPass>();
}

static PassRegistration<VectorizeMemRefPass> pass(
    "iree-spirv-vectorize-memref",
    "Vectorize memref arguments and allocations");
}  // namespace iree_compiler
}  // namespace mlir
