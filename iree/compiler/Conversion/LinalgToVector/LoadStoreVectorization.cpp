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

#include "iree/compiler/Conversion/LinalgToVector/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

constexpr int kVectorizationSizeInBits = 128;
constexpr int kVecSize = kVectorizationSizeInBits / (sizeof(float) * 8);

/// Returns a VectorType in `kVectorizationSizeInBits` bits if `t` is a scalar.
static VectorType getVecType(OpBuilder &builder, Type t) {
  if (!t.isa<IntegerType, FloatType>()) return {};
  if (t.getIntOrFloatBitWidth() != 32) return {};
  Type newElemType = t.isa<IntegerType>() ? builder.getI32Type().cast<Type>()
                                          : builder.getF32Type().cast<Type>();
  return VectorType::get(kVecSize, newElemType);
}

/// Returns the memref of vector converted from `type`.
static MemRefType getVectorizedMemRefType(OpBuilder &builder, MemRefType type) {
  Type elemType = type.getElementType();
  VectorType vecType = getVecType(builder, elemType);
  if (!vecType) return {};
  unsigned elemSize = elemType.getIntOrFloatBitWidth();
  unsigned vecSize = kVectorizationSizeInBits / elemSize;
  SmallVector<int64_t, 2> newShape(type.getShape().begin(),
                                   type.getShape().end());
  if (newShape.empty()) return {};
  if (newShape.back() % vecSize != 0) return {};
  newShape.back() = newShape.back() / vecSize;
  return MemRefType::get(newShape, vecType, {}, type.getMemorySpace());
}

/// Returns a vectorized `val`, ie, the result type is a VectorType.
static Value legalizeToVectorType(OpBuilder &builder, Value val) {
  Type type = val.getType();
  if (type.isa<VectorType>()) {
    return val;
  } else if (type.isIntOrFloat()) {
    auto vecType = getVecType(builder, type);
    if (!vecType) return nullptr;
    return builder.createOrFold<vector::BroadcastOp>(val.getLoc(), vecType,
                                                     val);
  }
  return nullptr;
}

/// Base class to vectorize std ops. If a generic op is vectorized, all the std
/// ops in the region should be vectorized as well.
///
/// This base class handles the check on operands and vectorization for all the
/// operands.
///
/// All derived classes implement a static apply method with the following
/// signature:
///
/// ```c++
/// LogicalResult apply(SrcOpTy op, ArrayRef<Value> args,
///                     ConversionPatternRewriter& rewriter) const;
/// ```
template <typename DerivedTy, typename SrcOpTy>
struct VectorizeOpBase : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    if (llvm::all_of(args, [](Value arg) {
          return arg.getType().isIntOrIndexOrFloat();
        })) {
      return failure();
    }
    SmallVector<Value, 4> vecArgs;
    for (Value arg : args) {
      Value val = legalizeToVectorType(rewriter, arg);
      if (!val) return failure();
      vecArgs.push_back(val);
    }
    return static_cast<DerivedTy const *>(this)->apply(op, vecArgs, rewriter);
  }
};

template <typename OpTy>
struct VectorizeElementwiseOp
    : public VectorizeOpBase<VectorizeElementwiseOp<OpTy>, OpTy> {
  using VectorizeOpBase<VectorizeElementwiseOp<OpTy>, OpTy>::VectorizeOpBase;
  LogicalResult apply(OpTy op, ArrayRef<Value> args,
                      ConversionPatternRewriter &rewriter) const {
    auto vecType = getVecType(rewriter, op.getResult().getType());
    if (!vecType) return failure();
    auto newOp = rewriter.create<OpTy>(op.getLoc(), vecType, args);
    rewriter.replaceOp(op, newOp.getOperation()->getResults());
    return success();
  }
};

template <typename OpTy>
struct VectorizeCmpOp : public VectorizeOpBase<VectorizeCmpOp<OpTy>, OpTy> {
  using VectorizeOpBase<VectorizeCmpOp<OpTy>, OpTy>::VectorizeOpBase;
  LogicalResult apply(OpTy op, ArrayRef<Value> args,
                      ConversionPatternRewriter &rewriter) const {
    auto newOp =
        rewriter.create<OpTy>(op.getLoc(), op.predicate(), args[0], args[1]);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct VectorizeSelectOp
    : public VectorizeOpBase<VectorizeSelectOp, mlir::SelectOp> {
  using VectorizeOpBase<VectorizeSelectOp, mlir::SelectOp>::VectorizeOpBase;
  LogicalResult apply(mlir::SelectOp op, ArrayRef<Value> args,
                      ConversionPatternRewriter &rewriter) const {
    auto newOp =
        rewriter.create<SelectOp>(op.getLoc(), args[0], args[1], args[2]);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct VectorizeGenericOp : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    // If a generic op does not take any input, it means it's working on
    // constants and those operations do not have canonicalization patterns to
    // fold it. For now just ignore to vectorize it.
    if (genericOp.getNumInputs() == 0) {
      return failure();
    }

    if (llvm::any_of(genericOp.iterator_types(), [](Attribute attr) {
          return attr.cast<StringAttr>().getValue() !=
                 getParallelIteratorTypeName();
        })) {
      return failure();
    }

    // Do not vectorize if one of the operand is 0-D or one of the operand is
    // not iterated on contiguous memory.
    for (auto map : genericOp.getIndexingMaps()) {
      if (map.getNumResults() == 0) return failure();
      AffineDimExpr innerMostExpr =
          map.getResults().back().dyn_cast<AffineDimExpr>();
      if (!innerMostExpr ||
          innerMostExpr.getPosition() != map.getNumDims() - 1) {
        return failure();
      }
    }

    SmallVector<IREE::PlaceholderOp, 4> operands;
    SmallVector<MemRefType, 4> vecMemRefs;
    for (auto operand : args) {
      auto op = operand.getDefiningOp<IREE::PlaceholderOp>();
      if (!op) return failure();
      if (!op.getOperation()->hasOneUse()) return failure();
      auto memrefType = op.getResult().getType().dyn_cast<MemRefType>();
      if (!memrefType) return failure();
      auto vecMemRef = getVectorizedMemRefType(rewriter, memrefType);
      if (!vecMemRef) return failure();
      operands.push_back(op);
      vecMemRefs.push_back(vecMemRef);
    }

    SmallVector<Value, 4> newArgs;
    for (auto it : llvm::zip(operands, vecMemRefs)) {
      IREE::PlaceholderOp placeholder = std::get<0>(it);
      MemRefType vecMemRef = std::get<1>(it);
      auto arg = rewriter.create<IREE::PlaceholderOp>(placeholder.getLoc(),
                                                      vecMemRef, ValueRange{},
                                                      placeholder.getAttrs());
      rewriter.replaceOp(placeholder, arg.getResult());
      newArgs.push_back(arg.getResult());
    }
    ArrayRef<Value> newArgsRef(newArgs.begin(), newArgs.end());
    auto newOp = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), genericOp.getResultTypes(),
        /*inputs=*/newArgsRef.take_front(genericOp.getNumInputs()),
        /*outputBuffers*/ newArgsRef.take_back(genericOp.getNumOutputs()),
        genericOp.indexing_mapsAttr(), genericOp.iterator_types(),
        /*doc=*/nullptr,
        /*library_call=*/nullptr, genericOp.sparseAttr());

    Region &newRegion = newOp.region();
    rewriter.inlineRegionBefore(genericOp.getRegion(), newRegion,
                                newRegion.end());
    Block &newBlock = newOp.region().front();
    TypeConverter::SignatureConversion signatureConverter(
        newBlock.getNumArguments());
    for (auto arg : llvm::enumerate(vecMemRefs)) {
      signatureConverter.addInputs(arg.index(), arg.value().getElementType());
    }
    rewriter.applySignatureConversion(&newOp.region(), signatureConverter);
    rewriter.replaceOp(genericOp, newOp.getResults());
    return success();
  }
};

struct LoadStoreVectorizationPass
    : public PassWrapper<LoadStoreVectorizationPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns;
    // clang-format off
    patterns.insert<
        VectorizeGenericOp,
        VectorizeCmpOp<CmpFOp>,
        VectorizeCmpOp<CmpIOp>,
        VectorizeSelectOp,
        VectorizeElementwiseOp<AbsFOp>,
        VectorizeElementwiseOp<AndOp>,
        VectorizeElementwiseOp<OrOp>,
        VectorizeElementwiseOp<XOrOp>,
        VectorizeElementwiseOp<AddFOp>,
        VectorizeElementwiseOp<AddIOp>,
        VectorizeElementwiseOp<CeilFOp>,
        VectorizeElementwiseOp<CosOp>,
        VectorizeElementwiseOp<DivFOp>,
        VectorizeElementwiseOp<ExpOp>,
        VectorizeElementwiseOp<FPExtOp>,
        VectorizeElementwiseOp<FPToSIOp>,
        VectorizeElementwiseOp<FPTruncOp>,
        VectorizeElementwiseOp<FloorFOp>,
        VectorizeElementwiseOp<LogOp>,
        VectorizeElementwiseOp<MulFOp>,
        VectorizeElementwiseOp<MulIOp>,
        VectorizeElementwiseOp<NegFOp>,
        VectorizeElementwiseOp<RemFOp>,
        VectorizeElementwiseOp<RsqrtOp>,
        VectorizeElementwiseOp<SIToFPOp>,
        VectorizeElementwiseOp<ShiftLeftOp>,
        VectorizeElementwiseOp<SignExtendIOp>,
        VectorizeElementwiseOp<SignedDivIOp>,
        VectorizeElementwiseOp<SignedShiftRightOp>,
        VectorizeElementwiseOp<SinOp>,
        VectorizeElementwiseOp<SqrtOp>,
        VectorizeElementwiseOp<SubFOp>,
        VectorizeElementwiseOp<SubIOp>,
        VectorizeElementwiseOp<TanhOp>,
        VectorizeElementwiseOp<TruncateIOp>,
        VectorizeElementwiseOp<UnsignedDivIOp>,
        VectorizeElementwiseOp<UnsignedRemIOp>,
        VectorizeElementwiseOp<UnsignedShiftRightOp>,
        VectorizeElementwiseOp<ZeroExtendIOp>>(context);
    // clang-format on

    ConversionTarget target(*context);
    // Mark vector dialect and plancholder op legal.
    target.addLegalDialect<vector::VectorDialect>();
    target.addLegalOp<IREE::PlaceholderOp>();

    // If a generic op is vectorized, it is legal.
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      if (!op.hasBufferSemantics()) return false;
      for (auto arg : op.getOperands()) {
        if (arg.getType()
                .cast<MemRefType>()
                .getElementType()
                .isSignlessIntOrFloat())
          return false;
      }
      return true;
    });

    // Mark all standard ops legal if they are operating on vector types.
    target.addDynamicallyLegalDialect<mlir::StandardOpsDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            [](Operation *op) {
              auto isVectorType = [](Type t) { return t.isa<VectorType>(); };
              return llvm::any_of(op->getOperandTypes(), isVectorType) ||
                     llvm::any_of(op->getResultTypes(), isVectorType);
            }));
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> createLoadStoreVectorizationPass() {
  return std::make_unique<LoadStoreVectorizationPass>();
}

static PassRegistration<LoadStoreVectorizationPass> pass(
    "iree-codegen-vectorize-linalg-ops", "Vectorize Linalg operations");

}  // namespace iree_compiler
}  // namespace mlir
