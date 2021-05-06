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

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Conversion/HLOToLinalg/HLOToLinalgOnTensorPasses.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// linalg.pad_tensor conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to convert a linalg.pad_tensor operation into a fill + subtensor
/// insert. This is needed till pad_tensor op can be fused with its consumers.
struct PadTensorOpConversion : public OpConversionPattern<linalg::PadTensorOp> {
  using OpConversionPattern<linalg::PadTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::PadTensorOp padTensorOp, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    linalg::PadTensorOpAdaptor padOpAdaptor(args,
                                            padTensorOp->getAttrDictionary());
    // Check that the region is just a yield operation which is returning a
    // scalar that is not one of the arguments of the linalg operation.
    Region &region = padTensorOp.region();
    Block &block = region.front();
    if (!llvm::hasSingleElement(block)) return failure();
    auto yieldOp = cast<linalg::YieldOp>(block.getTerminator());
    if (!llvm::hasSingleElement(yieldOp.values())) return failure();
    Value yieldVal = yieldOp.values().front();
    if (llvm::any_of(block.getArguments(),
                     [&](Value v) { return v == yieldVal; })) {
      return failure();
    }

    OpBuilder::InsertionGuard g(rewriter);
    Location loc = padTensorOp.getLoc();
    auto lowPad = padTensorOp.getMixedLowPad();
    auto highPad = padTensorOp.getMixedHighPad();
    Value source = padOpAdaptor.source();
    RankedTensorType sourceType = padTensorOp.getSourceType();
    int64_t rank = sourceType.getRank();

    // TODO(ravishankarm): Use shape inference interface to get this.
    SmallVector<OpFoldResult> sourceShape;
    SmallVector<OpFoldResult> outputShape;
    for (int64_t dim : llvm::seq<int64_t>(0, rank)) {
      SmallVector<Value> mapValues;
      Value sourceDim = rewriter.createOrFold<memref::DimOp>(loc, source, dim);
      mapValues.push_back(sourceDim);
      sourceShape.push_back(sourceDim);
      AffineExpr expr = rewriter.getAffineDimExpr(0);
      unsigned numSymbols = 0;
      auto addValueOrAttr = [&](AffineExpr e, OpFoldResult valueOrAttr) {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          e = e + attr.cast<IntegerAttr>().getInt();
          return e;
        }
        e = e + rewriter.getAffineSymbolExpr(numSymbols++);
        mapValues.push_back(valueOrAttr.get<Value>());
        return e;
      };
      expr = addValueOrAttr(expr, lowPad[dim]);
      expr = addValueOrAttr(expr, highPad[dim]);
      Value v = linalg::applyMapToValues(
          rewriter, loc, AffineMap::get(1, numSymbols, expr), mapValues)[0];
      if (auto cst = v.getDefiningOp<ConstantOp>()) {
        outputShape.push_back(cst.value());
      } else {
        outputShape.push_back(v);
      }
    }
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, outputShape, sourceType.getElementType());
    Value fill =
        rewriter.create<linalg::FillOp>(loc, initTensor, yieldVal).getResult(0);
    SmallVector<OpFoldResult> strides(rank, rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<SubTensorInsertOp>(
        padTensorOp, source, fill, lowPad, sourceShape, strides);
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.concatenate conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.concatenate operation to subtensor ops + subtensor_insert ops.
struct ConcatenateOpConversion
    : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = this->typeConverter->convertType(op.getResult().getType())
                          .dyn_cast<RankedTensorType>();
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static shape for output");
    }

    Location loc = op.getLoc();
    int dim = op.dimension();
    int rank = resultType.getRank();
    SmallVector<Value, 3> offsets, sizes, strides;
    for (int i = 0; i < rank; ++i) {
      offsets.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
      sizes.push_back(rewriter.create<memref::DimOp>(loc, args[0], i));
      strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
    }
    Value resultDimSize = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      auto size = rewriter.create<memref::DimOp>(loc, arg, dim);
      resultDimSize = rewriter.create<AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    Value result =
        rewriter.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

    Value accBound = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      offsets[dim] = accBound;
      sizes[dim] = rewriter.create<memref::DimOp>(loc, arg, dim);
      result = rewriter.create<SubTensorInsertOp>(loc, arg, result, offsets,
                                                  sizes, strides);
      accBound = rewriter.create<AddIOp>(loc, accBound, sizes[dim]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.fft conversion patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Creats coefficients based on DFT definition, see
/// https://en.wikipedia.org/wiki/Discrete_Fourier_transform
Value getDFTMatmulCoeff(OpBuilder b, Location loc, RankedTensorType matrixType,
                        bool isRealPart) {
  // scale = 2 * pi / N
  double scale = 2 * M_PI / matrixType.getDimSize(0);

  SmallVector<Attribute> values;
  assert(matrixType.getRank() == 2 && "expected 2D matrix");
  for (auto i : llvm::seq<unsigned>(0, matrixType.getDimSize(0))) {
    for (auto j : llvm::seq<unsigned>(0, matrixType.getDimSize(1))) {
      double v = scale * i * j;
      if (isRealPart) {
        v = cos(v);
      } else {
        v = -sin(v);
      }
      values.push_back(b.getF32FloatAttr(v));
    }
  }
  return b.create<ConstantOp>(loc, matrixType,
                              DenseFPElementsAttr::get(matrixType, values));
}

Value createLinalgMatmulOnTensors(OpBuilder b, Location loc,
                                  RankedTensorType resultType, Value lhs,
                                  Value rhs) {
  Value zero =
      b.create<ConstantOp>(loc, b.getZeroAttr(resultType.getElementType()));
  auto initTensor = b.create<linalg::InitTensorOp>(
      loc, /*dyn_size=*/ValueRange{}, resultType.getShape(),
      resultType.getElementType());
  Value zeroTensor =
      b.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

  switch (lhs.getType().cast<RankedTensorType>().getRank()) {
    case 1:
      return b
          .create<linalg::VecmatOp>(loc, TypeRange{resultType},
                                    ValueRange{lhs, rhs},
                                    ValueRange{zeroTensor})
          .getResult(0);
    case 2:
      return b
          .create<linalg::MatmulOp>(loc, TypeRange{resultType},
                                    ValueRange{lhs, rhs},
                                    ValueRange{zeroTensor})
          .getResult(0);
    default:
      llvm_unreachable("unhandled matmul type");
  }
}

/// Converts mhlo.fft operation to Linalg ops.
struct FftOpConversion : public OpConversionPattern<mhlo::FftOp> {
  using OpConversionPattern<mhlo::FftOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::FftOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    if (op.fft_type() != "RFFT") {
      return rewriter.notifyMatchFailure(op,
                                         "non RFFT types are supported yet");
    }

    mhlo::FftOpAdaptor adaptor(args);
    auto inputType = adaptor.operand().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() > 2) {
      return rewriter.notifyMatchFailure(op, "only static 1D or 2D dft ops");
    }

    int rank = inputType.getRank();
    int n = inputType.getDimSize(rank - 1);
    int fftLength =
        op.fft_length().getSplatValue().cast<IntegerAttr>().getInt() / 2 + 1;

    Location loc = op.getLoc();
    auto matrixType =
        RankedTensorType::get({n, fftLength}, inputType.getElementType());
    auto resultType =
        RankedTensorType::get(op.getType().cast<RankedTensorType>().getShape(),
                              inputType.getElementType());

    auto realMatrix =
        getDFTMatmulCoeff(rewriter, loc, matrixType, /*isRealPart=*/true);
    auto real = createLinalgMatmulOnTensors(rewriter, loc, resultType,
                                            adaptor.operand(), realMatrix);

    auto imagMatrix =
        getDFTMatmulCoeff(rewriter, loc, matrixType, /*isRealPart=*/false);
    auto imag = createLinalgMatmulOnTensors(rewriter, loc, resultType,
                                            adaptor.operand(), imagMatrix);

    // Pack the results back to mhlo::ComplexOp.
    rewriter.replaceOpWithNewOp<mhlo::ComplexOp>(op, op.getType(), real, imag);
    return success();
  }
};
}  // namespace

struct ConvertHLOToLinalgOnTensorsPass
    : public PassWrapper<ConvertHLOToLinalgOnTensorsPass, FunctionPass> {
  ConvertHLOToLinalgOnTensorsPass(bool useLinalgOnTensorsPath = false)
      : useLinalgOnTensorsPath(useLinalgOnTensorsPath){};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, linalg::LinalgDialect,
                    mhlo::MhloDialect, ShapeDialect, math::MathDialect,
                    memref::MemRefDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    MLIRContext *context = &getContext();

    auto typeConverter = mhlo::createHloToLinalgSignedIntegerConverter();
    populateHLOToLinalgOnTensorsConversionPatterns(context, *typeConverter,
                                                   patterns);
    if (useLinalgOnTensorsPath) {
      patterns.insert<PadTensorOpConversion>(context);
    }

    ConversionTarget target(getContext());
    target.addIllegalDialect<mhlo::MhloDialect>();

    // TODO(hanchung): Do it in a cleaner way.
    // We don't see complex types in codegen. This assumes that we run
    // LowerComplexPass before, and all the complex types were folded away.
    // Mark them legal and rely on canonicalization patterns to fold them away.
    target.addDynamicallyLegalOp<mhlo::ComplexOp>(
        [](mhlo::ComplexOp op) { return true; });
    target.addDynamicallyLegalOp<mhlo::RealOp>(
        [](mhlo::RealOp op) { return true; });
    target.addDynamicallyLegalOp<mhlo::ImagOp>(
        [](mhlo::ImagOp op) { return true; });

    // Let the rest fall through.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (useLinalgOnTensorsPath) {
      // Set linalg.pad_tensor illegal for now.
      target.addIllegalOp<linalg::PadTensorOp>();
    }

    if (failed(applyPartialConversion(getFunction(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }

 private:
  bool useLinalgOnTensorsPath;
};

/// This pass is just added for lit-testing when using the linalg on tensors
/// path. Remove when the linalg on tensors path becomes default.
struct ConvertHLOToLinalgOnTensorsPassExperimental
    : public ConvertHLOToLinalgOnTensorsPass {
  ConvertHLOToLinalgOnTensorsPassExperimental()
      : ConvertHLOToLinalgOnTensorsPass(true) {}
};

/// Convert mhlo.constant op into std.const.
struct ConstOpConversion : public OpConversionPattern<mhlo::ConstOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ConstOp op, ArrayRef<Value> /*operands*/,
      ConversionPatternRewriter &rewriter) const override {
    auto valueAttr = op.value();
    Type oldElType = valueAttr.getType().getElementType();
    Type newElType = this->typeConverter->convertType(oldElType);
    ElementsAttr newValueAttr = valueAttr;
    if (newElType != oldElType) {
      // Values don't change, just their reported type.
      newValueAttr = valueAttr.mapValues(
          newElType, [](const APInt &oldEl) { return oldEl; });
    }
    rewriter.replaceOpWithNewOp<ConstantOp>(op, newValueAttr);
    return success();
  }
};

}  // namespace

void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  mhlo::populateHLOToLinalgConversionPattern(context, typeConverter, &patterns);
  patterns.insert<ConstOpConversion, ConcatenateOpConversion, FftOpConversion>(
      typeConverter, context);
}

static llvm::cl::opt<bool> clUseLinalgOnTensorsPath(
    "iree-linalg-on-tensors-path",
    llvm::cl::desc("Convert from MHLO to Linalg on tensors for linalg on "
                   "tensor codegen path"),
    llvm::cl::init(false));

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass(
    bool useLinalgOnTensorsPath) {
  return std::make_unique<ConvertHLOToLinalgOnTensorsPass>(
      useLinalgOnTensorsPath);
}

static PassRegistration<ConvertHLOToLinalgOnTensorsPass> legalize_pass(
    "iree-codegen-hlo-to-linalg-on-tensors",
    "Convert from XLA-HLO ops to Linalg ops on tensors", []() {
      return std::make_unique<ConvertHLOToLinalgOnTensorsPass>(
          clUseLinalgOnTensorsPath);
    });

}  // namespace iree_compiler
}  // namespace mlir
