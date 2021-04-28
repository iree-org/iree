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
// This file implements a pass to vectorize scalar interface memrefs into vector
// ones in order to perform vector load/store on these memrefs to achieve better
// memory access patterns.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

constexpr int kMaxVectorNumBits = 128;
constexpr int kMaxVectorNumElements = 4;

namespace mlir {
namespace iree_compiler {

/// Writes all uses of the given memref `value` and returns true if all uses are
/// transfer read/write operations.
static bool getUsesIfAllTransferOp(Value value,
                                   SmallVectorImpl<Operation *> &uses) {
  assert(uses.empty() && "expected uses to be empty");
  for (Operation *userOp : value.getUsers()) {
    if (isa<memref::DeallocOp>(userOp)) continue;
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
  }
  if (type.isa<VectorType>()) {
    auto vecType = type.cast<VectorType>();
    auto elementType = vecType.getElementType();
    return elementType.getIntOrFloatBitWidth() * vecType.getNumElements();
  }
  return {};
}

// Calculates the vector size we want to use based on the memref uses.
static unsigned calculateMemrefVecSize(SmallVectorImpl<Operation *> &uses) {
  unsigned minSize = kMaxVectorNumBits;
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
static unsigned isMemRefAndVectorizable(Value value,
                                        SmallVectorImpl<Operation *> &uses) {
  auto memrefType = value.getType().dyn_cast<MemRefType>();

  // Require scalar element type
  if (!memrefType || memrefType.getElementType().isa<VectorType>()) return 0;

  // Require static innermost dimension.
  if (memrefType.getRank() == 0 ||
      ShapedType::isDynamic(memrefType.getShape().back()))
    return 0;

  // If we have an odd number of elements, it will require padding in the
  // buffer.
  if (memrefType.getShape().back() % 2 != 0) return 0;

  if (kMaxVectorNumBits % memrefType.getElementTypeBitWidth() != 0) return 0;

  if (getUsesIfAllTransferOp(value, uses)) return calculateMemrefVecSize(uses);
  return 0;
}

namespace {
/// Analyze memref usages to decide if it should be vectorized. Right now the
/// logic is to vectorize memref only if it is used by vector transfer
/// read/write ops.
class MemRefUsageAnalysis {
 public:
  explicit MemRefUsageAnalysis(mlir::Operation *);

  // Returns true if the memref should be converted to a vector of memref.
  bool vectorizeMemRef(Value value) const {
    return valueToVectorBitsMap.count(value);
  }

  // Return the size of the vector we want to use for memref vectorization.
  unsigned getMemRefVectorSizeInBits(Value value) const {
    return valueToVectorBitsMap.find(value)->second;
  }
  // Returns true if the transfer operation needs to be updated during memref
  // vectorization.
  bool transferConvert(Operation *op) const { return transferOps.count(op); }

 private:
  void analyzeMemRefValue(Value value);

  llvm::DenseMap<Value, unsigned> valueToVectorBitsMap;
  llvm::DenseSet<Operation *> transferOps;
};

MemRefUsageAnalysis::MemRefUsageAnalysis(mlir::Operation *op) {
  op->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<FuncOp>([this](FuncOp funcOp) {
          for (Value arg : funcOp.getArguments()) {
            analyzeMemRefValue(arg);
          }
        })
        .Case<memref::AllocOp, IREE::PlaceholderOp,
              IREE::HAL::InterfaceBindingSubspanOp>(
            [this](auto op) { analyzeMemRefValue(op); });
  });
}

void MemRefUsageAnalysis::analyzeMemRefValue(Value value) {
  SmallVector<Operation *, 4> vectorUses;
  if (unsigned vectorSize = isMemRefAndVectorizable(value, vectorUses)) {
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
    if (!memrefUsageAnalysis.transferConvert(read)) {
      return rewriter.notifyMatchFailure(
          read, "cannot be vectorized per memref usage analysis");
    }

    Location loc = read.getLoc();
    vector::TransferReadOp::Adaptor adaptor(operands,
                                            read->getAttrDictionary());

    auto scalarMemrefType = read.source().getType().dyn_cast<MemRefType>();
    auto vectorMemrefType = adaptor.source().getType().dyn_cast<MemRefType>();
    auto readVectorType = read.getVectorType();
    if (!scalarMemrefType || !vectorMemrefType) return failure();

    Optional<unsigned> vectorMemrefElemSize =
        getBitWidth(vectorMemrefType.getElementType());
    Optional<unsigned> scalarMemrefElemSize =
        getBitWidth(scalarMemrefType.getElementType());
    Optional<unsigned> readVecSize = getBitWidth(readVectorType);
    if (!vectorMemrefElemSize || !scalarMemrefElemSize || !readVecSize)
      return failure();

    unsigned ratio = *vectorMemrefElemSize / *scalarMemrefElemSize;
    SmallVector<Value, 4> indices(adaptor.indices().begin(),
                                  adaptor.indices().end());
    indices.back() = rewriter.create<SignedDivIOp>(
        loc, indices.back(), rewriter.create<ConstantIndexOp>(loc, ratio));

    // If the transfer_read can be replaced by a load after vectorization use
    // LoadOp and cast back to the original type.
    if (*vectorMemrefElemSize == *readVecSize) {
      Type elemType = vectorMemrefType.getElementType();
      Value newLoad = rewriter.create<memref::LoadOp>(
          loc, elemType, adaptor.source(), indices);
      Type serializedVecType =
          VectorType::get(read.getVectorType().getNumElements(),
                          read.getVectorType().getElementType());
      newLoad =
          rewriter.create<vector::BitCastOp>(loc, serializedVecType, newLoad);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
          read, read.getVectorType(), newLoad);
    } else {
      rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
          read, read.getVectorType(), adaptor.source(), indices);
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
    if (!memrefUsageAnalysis.transferConvert(write)) {
      return rewriter.notifyMatchFailure(
          write, "cannot be vectorized per memref usage analysis");
    }

    Location loc = write.getLoc();
    vector::TransferWriteOp::Adaptor adaptor(operands,
                                             write->getAttrDictionary());

    auto scalarMemrefType = write.source().getType().dyn_cast<MemRefType>();
    auto vectorMemrefType = adaptor.source().getType().dyn_cast<MemRefType>();
    auto writeVectorType = write.getVectorType();
    if (!scalarMemrefType || !vectorMemrefType) return failure();

    Optional<unsigned> vectorMemrefElemSize =
        getBitWidth(vectorMemrefType.getElementType());
    Optional<unsigned> scalarMemrefElemSize =
        getBitWidth(scalarMemrefType.getElementType());
    Optional<unsigned> writeVecSize = getBitWidth(writeVectorType);
    if (!vectorMemrefElemSize || !scalarMemrefElemSize || !writeVecSize)
      return failure();

    unsigned ratio = *vectorMemrefElemSize / *scalarMemrefElemSize;
    SmallVector<Value, 4> indices(adaptor.indices());
    indices.back() = rewriter.create<SignedDivIOp>(
        loc, indices.back(), rewriter.create<ConstantIndexOp>(loc, ratio));

    // If the transfer_write can be replaced by a store after vectorization cast
    // the original value and use StoreOp.
    if (*vectorMemrefElemSize == *writeVecSize) {
      Type serializedVecType = VectorType::get(
          writeVectorType.getNumElements(), writeVectorType.getElementType());
      Value data = rewriter.create<vector::ShapeCastOp>(loc, serializedVecType,
                                                        adaptor.vector());
      data = rewriter.create<vector::BitCastOp>(
          loc, vectorMemrefType.getElementType(), data);
      rewriter.replaceOpWithNewOp<memref::StoreOp>(write, data,
                                                   adaptor.source(), indices);
    } else {
      rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
          write, adaptor.vector(), adaptor.source(), indices);
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
  return MemRefType::get(newShape, vecType, {}, type.getMemorySpaceAsInt());
}

class ProcessAlloc final : public MemRefConversionPattern<memref::AllocOp> {
 public:
  using MemRefConversionPattern<memref::AllocOp>::MemRefConversionPattern;
  LogicalResult matchAndRewrite(
      memref::AllocOp alloc, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto memrefType = getVectorizedMemRefType(rewriter, alloc.getResult());
    if (!memrefType) return failure();
    rewriter.replaceOpWithNewOp<memref::AllocOp>(alloc, *memrefType,
                                                 alloc.dynamicSizes());
    return success();
  }
};

class ProcessPlaceHolder final
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
    rewriter.replaceOpWithNewOp<IREE::PlaceholderOp>(
        placeholder, *vecMemRef, ValueRange(), placeholder->getAttrs());
    return success();
  }
};

class ProcessInterfaceBinding final
    : public MemRefConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
 public:
  using MemRefConversionPattern<
      IREE::HAL::InterfaceBindingSubspanOp>::MemRefConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp bindingOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto memrefType = bindingOp.getType().dyn_cast<MemRefType>();
    if (!memrefType) return failure();
    auto vecMemRef = getVectorizedMemRefType(rewriter, bindingOp.getResult());
    if (!vecMemRef) return failure();
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        bindingOp, *vecMemRef, bindingOp.binding(), bindingOp.byte_offset(),
        bindingOp.byte_length());
    return success();
  }
};

/// Translates vector.transfer_read with less than 4 scalars into reading each
/// scalar and then compose the vector.
///
/// This is a very specific pattern for handling corner cases and boundary
/// cases. For example, in vision models we can have the initial image with
/// three channels. We cannot perform the native load4 there; by performing
/// scalar read we lose some benefits of load4 but we can still make sure the
/// overall vectorization does not fail.
struct ScalarizeVectorTransferRead final
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = readOp.getType();
    Type scalarType = vectorType.getElementType();
    if (vectorType.getRank() != 1 || vectorType.getDimSize(0) >= 4)
      return failure();

    Location loc = readOp.getLoc();

    SmallVector<Value, 4> scalars;
    SmallVector<Value, 4> indices(readOp.indices().begin(),
                                  readOp.indices().end());
    for (int i = 0; i < vectorType.getDimSize(0); ++i) {
      indices.back() = rewriter.createOrFold<ConstantIndexOp>(loc, i);
      scalars.push_back(rewriter.create<memref::LoadOp>(
          loc, scalarType, readOp.source(), indices));
    }

    Value newVector = rewriter.create<ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));
    for (int i = 0; i < vectorType.getDimSize(0); ++i) {
      Value index = rewriter.createOrFold<ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
      newVector = rewriter.create<vector::InsertElementOp>(loc, scalars[i],
                                                           newVector, index);
    }

    rewriter.replaceOp(readOp, newVector);

    return success();
  }
};

class VectorizeMemRefLoadStorePass final
    : public PassWrapper<VectorizeMemRefLoadStorePass,
                         OperationPass<ModuleOp>> {
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

void VectorizeMemRefLoadStorePass::runOnOperation() {
  // Uses the signature conversion methodology of the dialect conversion
  // framework to implement the conversion.
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  memrefUsageAnalysis = &getAnalysis<MemRefUsageAnalysis>();

  RewritePatternSet conversionPatterns(context);
  conversionPatterns
      .add<ProcessFuncArg, ProcessTransferRead, ProcessTransferWrite,
           ProcessAlloc, ProcessPlaceHolder, ProcessInterfaceBinding>(
          context, *memrefUsageAnalysis);

  ConversionTarget target(*context);
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return llvm::all_of(op.getArguments(), [&](Value arg) {
      return !memrefUsageAnalysis->vectorizeMemRef(arg);
    });
  });
  target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp alloc) {
    return !memrefUsageAnalysis->vectorizeMemRef(alloc);
  });
  target.addDynamicallyLegalOp<IREE::PlaceholderOp>(
      [&](IREE::PlaceholderOp placeholder) {
        return !memrefUsageAnalysis->vectorizeMemRef(placeholder);
      });
  target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
      [&](IREE::HAL::InterfaceBindingSubspanOp bindingOp) {
        return !memrefUsageAnalysis->vectorizeMemRef(bindingOp);
      });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    if (isa<vector::TransferWriteOp, vector::TransferReadOp>(op))
      return !memrefUsageAnalysis->transferConvert(op);
    return true;
  });
  if (failed(applyPartialConversion(module, target,
                                    std::move(conversionPatterns))))
    return signalPassFailure();

  for (FuncOp func : module.getOps<FuncOp>()) {
    RewritePatternSet rewritingPatterns(context);
    rewritingPatterns.add<ScalarizeVectorTransferRead>(context);

    (void)applyPatternsAndFoldGreedily(func, std::move(rewritingPatterns));
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createVectorizeMemrefLoadStorePass() {
  return std::make_unique<VectorizeMemRefLoadStorePass>();
}

static PassRegistration<VectorizeMemRefLoadStorePass> pass(
    "iree-spirv-vectorize-memref-load-store",
    "Vectorize interface memrefs and their load/store for better memory "
    "access");
}  // namespace iree_compiler
}  // namespace mlir
