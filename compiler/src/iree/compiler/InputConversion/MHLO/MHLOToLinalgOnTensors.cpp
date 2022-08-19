// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/InputConversion/MHLO/ConvertMHLOToFlow.h"
#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Rewriters.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/legalize_to_linalg_utils.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
namespace MHLO {

//===----------------------------------------------------------------------===//
// mhlo.concatenate conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.concatenate operation to extract_slice ops + insert_slice ops.
struct ConcatenateOpConversion
    : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, OpAdaptor adaptor,
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
      offsets.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      sizes.push_back(rewriter.createOrFold<tensor::DimOp>(
          loc, adaptor.getOperands()[0], i));
      strides.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }
    Value resultDimSize = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto arg : adaptor.getOperands()) {
      auto size = rewriter.createOrFold<tensor::DimOp>(loc, arg, dim);
      resultDimSize =
          rewriter.createOrFold<arith::AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;
    Value result = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto toOpFoldResult = [](Value v) -> OpFoldResult {
      auto op = v.getDefiningOp<arith::ConstantIndexOp>();
      if (!op) return v;
      return op.getValue();
    };

    Value accBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto arg : adaptor.getOperands()) {
      offsets[dim] = accBound;
      sizes[dim] = rewriter.createOrFold<tensor::DimOp>(loc, arg, dim);
      result = rewriter.create<tensor::InsertSliceOp>(
          loc, arg, result,
          llvm::to_vector(llvm::map_range(offsets, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(sizes, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(strides, toOpFoldResult)));
      accBound = rewriter.create<arith::AddIOp>(loc, accBound, sizes[dim]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// mhlo.fft conversion patterns.
//===----------------------------------------------------------------------===//

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
  return b.create<arith::ConstantOp>(
      loc, matrixType, DenseFPElementsAttr::get(matrixType, values));
}

Value createLinalgMatmulOnTensors(OpBuilder b, Location loc,
                                  RankedTensorType resultType, Value lhs,
                                  Value rhs) {
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getZeroAttr(resultType.getElementType()));
  Value initTensor = b.create<linalg::InitTensorOp>(
      loc, /*dyn_size=*/ValueRange{}, resultType.getShape(),
      resultType.getElementType());
  Value zeroTensor =
      b.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

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
      assert(false && "unhandled matmul type");
      return Value();
  }
}

/// Converts mhlo.fft operation to Linalg ops.
struct FftOpConversion : public OpConversionPattern<mhlo::FftOp> {
  using OpConversionPattern<mhlo::FftOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::FftOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.fft_type() != mhlo::FftType::RFFT) {
      return rewriter.notifyMatchFailure(op,
                                         "non RFFT types are supported yet");
    }

    auto inputType = adaptor.operand().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() > 2) {
      return rewriter.notifyMatchFailure(op, "only static 1D or 2D dft ops");
    }

    int rank = inputType.getRank();
    int n = inputType.getDimSize(rank - 1);
    int fftLength =
        op.fft_length().getSplatValue<IntegerAttr>().getInt() / 2 + 1;

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

// TODO(#9361): Retire the pattern. The implementation is partial, and it has
// bad performance. It's added for not regressing scatter support.
struct ScatterUpdateConversion : public OpConversionPattern<mhlo::ScatterOp> {
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Variadic Scatter support not yet implemented
    if (op.operands().size() != 1 || op.updates().size() != 1) return failure();

    // Check if it is a tensor_scatter_nd_update-like op.
    if (op.getRegion().front().getNumArguments() != 2) return failure();

    auto operandTy =
        adaptor.operands()[0].getType().dyn_cast<RankedTensorType>();
    auto indicesTy =
        adaptor.scatter_indices().getType().dyn_cast<RankedTensorType>();
    if (!operandTy || !indicesTy) return failure();

    // Linalg operations put all the computation to the innermost loop. Since we
    // also iterate over scatter_indices() with some loops, we can only check
    // one scatter index in one iteration. If there are multiple indices (ie,
    // the index depth is greater than 1), we don't have a way to keep the
    // comparison state. E.g., if the index_depth is 2, like indices = [[0, 1]],
    // we should use the update value only if (i == 0 and j == 1). However, we
    // can not get both indices in one iteration unless we pack them together.
    auto indexVectorDim = op.scatter_dimension_numbers().getIndexVectorDim();
    if (indicesTy.getDimSize(indexVectorDim) != 1)
      return rewriter.notifyMatchFailure(op, "require index depth to be 1");
    if (indexVectorDim != indicesTy.getRank() - 1) {
      return rewriter.notifyMatchFailure(
          op, "require index_vector_dim to be the last dim");
    }

    // One of indices dims is index depth vector.
    int64_t nloops = operandTy.getRank() + indicesTy.getRank() - 1;
    SmallVector<AffineMap, 3> indexingMaps;
    {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = 0, e = operandTy.getRank(); i < e; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      indexingMaps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                            rewriter.getContext()));
    }
    {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = operandTy.getRank(); i < nloops; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      // The index depth is 1.
      exprs.push_back(rewriter.getAffineConstantExpr(0));
      indexingMaps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                            rewriter.getContext()));

      exprs.pop_back();
      auto updateWindowDims =
          op.scatter_dimension_numbers().getUpdateWindowDims();
      for (auto d : updateWindowDims)
        exprs.push_back(rewriter.getAffineDimExpr(d));
      indexingMaps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                            rewriter.getContext()));
    }
    indexingMaps.push_back(indexingMaps.front());

    auto resultTy =
        this->typeConverter->convertType(op.getResults()[0].getType())
            .cast<ShapedType>();
    auto scatterDimsToOperandDims =
        op.scatter_dimension_numbers().getScatterDimsToOperandDims();
    assert(scatterDimsToOperandDims.size() == 1);
    // Do not need init_tensor because we'd like to initialize the output as
    // operand.
    auto loc = op.getLoc();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/ArrayRef<Type>{resultTy},
        /*inputs=*/
        ValueRange{adaptor.operands()[0], adaptor.scatter_indices(),
                   adaptor.updates()[0]},
        /*outputs=*/adaptor.operands()[0], indexingMaps,
        mhlo::getNParallelLoopsAttrs(nloops),
        [](OpBuilder &b, Location loc, ValueRange args) {},
        mhlo::pruneAttributeList(op));

    // Transform the scatter update computation region
    //   update = a bunch of computation
    //   return update
    // to linalg.generic region:
    //   update = a bunch of computation
    //   result = idx == cmpIdx ? update : old_value
    //   linalg.yield result
    bool updateIsTrivial = (op.getRegion().front().getOperations().size() == 1);
    Block *block = &linalgOp->getRegion(0).front();
    auto args = block->getArguments();

    BlockAndValueMapping mapping;
    // The scatter update computation block arguments are tensors of scalars
    // while the linalg.generic block arguments are scalars.
    if (updateIsTrivial) {
      // If there is no actual update computation, directly use the
      // linalg.generic block arguments.
      for (auto pair : llvm::zip_first(op.getRegion().front().getArguments(),
                                       args.drop_front(2)))
        mapping.map(std::get<0>(pair), std::get<1>(pair));
    } else {
      // Otherwise, convert the linalg.generic block scalar arguments to
      // tensors, to avoid producing illegal mhlo instructions.
      rewriter.setInsertionPointToStart(block);
      for (auto pair : llvm::zip_first(op.getRegion().front().getArguments(),
                                       args.drop_front(2)))
        mapping.map(std::get<0>(pair),
                    rewriter.create<tensor::FromElementsOp>(
                        loc, std::get<0>(pair).getType(), std::get<1>(pair)));
    }

    // Transform the computation block over to the linalg.generic op.
    rewriter.cloneRegionBefore(op.getRegion(), linalgOp->getRegion(0),
                               linalgOp->getRegion(0).end(), mapping);
    rewriter.mergeBlocks(&linalgOp->getRegion(0).back(), block, llvm::None);

    // Generate: result = idx == cmpIdx ? update : old_value.
    Operation *terminator = block->getTerminator();
    rewriter.setInsertionPoint(terminator);
    Value cmpIdx =
        rewriter.create<linalg::IndexOp>(loc, scatterDimsToOperandDims[0]);
    Value idx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), args[1]);
    Value pred = rewriter.create<arith::CmpIOp>(
        loc, rewriter.getI1Type(), arith::CmpIPredicate::eq, cmpIdx, idx);
    Value result = terminator->getOperand(0);
    if (!updateIsTrivial)
      result = rewriter.create<tensor::ExtractOp>(loc, result, ValueRange({}));
    result = rewriter.create<arith::SelectOp>(loc, args[2].getType(), pred,
                                              args[2], result);

    op.emitWarning("op is lowered to an inefficient way, which is unexpected");
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(terminator, result);
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

// We need to convert func ops in order to convert types.
class BuiltinFuncOpPattern : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::FuncOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());

    // Convert function arguments.
    for (unsigned i = 0, e = srcFuncType.getNumInputs(); i < e; ++i) {
      if (failed(getTypeConverter()->convertSignatureArg(
              i, srcFuncType.getInput(i), signatureConversion))) {
        return rewriter.notifyMatchFailure(srcOp, "argument failed to convert");
      }
    }

    // Convert function results.
    SmallVector<Type> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(srcFuncType.getResults(),
                                                convertedResultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }

    // Create new function with converted argument and result types.
    auto newFuncType = mlir::FunctionType::get(
        srcOp.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);

    // Update the function in place.
    rewriter.startRootUpdate(srcOp);
    srcOp.setType(newFuncType);

    // Tell the rewriter to convert the region signature.
    TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&srcOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.finalizeRootUpdate(srcOp);
    return success();
  }
};

class GenericTypeConvert : public ConversionPattern {
 public:
  GenericTypeConvert(StringRef rootName, TypeConverter &converter,
                     MLIRContext *context, PatternBenefit benefit = 0)
      : ConversionPattern(converter, rootName, benefit, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute, 4> newAttr;
    llvm::append_range(newAttr, op->getAttrs());
    llvm::SmallVector<Type, 4> newResults;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResults))) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttr, op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      if (failed(getTypeConverter()->convertSignatureArgs(
              newRegion->getArgumentTypes(), result))) {
        return rewriter.notifyMatchFailure(op,
                                           "argument type conversion failed");
      }
      rewriter.applySignatureConversion(newRegion, result);
    }
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

llvm::Optional<Value> scalarToTensor(OpBuilder &builder, Type /*type*/,
                                     ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  if (inputs.front().getType().isa<ShapedType>()) {
    return llvm::None;
  }
  return builder
      .create<tensor::FromElementsOp>(
          loc, RankedTensorType::get({}, inputs.front().getType()),
          inputs.front())
      .getResult();
}

struct ConvertMHLOToLinalgOnTensorsPass
    : public ConvertMHLOToLinalgOnTensorsBase<
          ConvertMHLOToLinalgOnTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, linalg::LinalgDialect,
                    mhlo::MhloDialect, shape::ShapeDialect, math::MathDialect,
                    memref::MemRefDialect, complex::ComplexDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    MLIRContext *context = &getContext();

    auto typeConverter = mhlo::createHloToLinalgTypeConverter();
    typeConverter->addArgumentMaterialization(scalarToTensor);
    // NOTE: not using corresponding setupMHLOToFlowPatterns because the entire
    // MHLO dialects are marked illegal by this pass.
    // TODO: Collapse/rework all of these patterns once the consolidation
    // lands. There is little reason to have these so spread out.
    populateMHLOToFlowPatterns(context, patterns);
    chlo::populateDecomposeChloPatterns(context, &patterns);
    populateMHLOBroadcastingToLinalgPatterns(context, *typeConverter, patterns);
    populateMHLOToLinalgOnTensorsConversionPatterns(context, *typeConverter,
                                                    patterns);
    populateMHLOComplexToRealPatterns(context, *typeConverter, patterns);

    // Structural patterns (functions, cfg, terminators).
    patterns.insert<BuiltinFuncOpPattern>(*typeConverter, context);
    patterns.insert<GenericTypeConvert>(func::ReturnOp::getOperationName(),
                                        *typeConverter, context);
    patterns.insert<GenericTypeConvert>(func::CallOp::getOperationName(),
                                        *typeConverter, context);
    patterns.insert<GenericTypeConvert>(cf::CondBranchOp::getOperationName(),
                                        *typeConverter, context);
    patterns.insert<GenericTypeConvert>(cf::BranchOp::getOperationName(),
                                        *typeConverter, context);

    ConversionTarget target(getContext());
    auto isIllegalType = [&](Type t) { return !typeConverter->isLegal(t); };
    auto isLegallyTypedOp = [&](Operation *op) -> bool {
      for (Type type : op->getResultTypes()) {
        if (isIllegalType(type)) return false;
      }
      for (Type type : op->getOperandTypes()) {
        if (isIllegalType(type)) return false;
      }
      return true;
    };

    target.addIllegalDialect<chlo::ChloDialect>();
    target.addIllegalDialect<mhlo::MhloDialect>();

    // Functions must have legal types.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp funcOp) {
      for (Type type : funcOp.getFunctionType().getInputs()) {
        if (isIllegalType(type)) return false;
      }
      for (Type type : funcOp.getFunctionType().getResults()) {
        if (isIllegalType(type)) return false;
      }
      for (Block &block : funcOp.getBody()) {
        for (Type type : block.getArgumentTypes()) {
          if (isIllegalType(type)) return false;
        }
      }
      return true;
    });

    // Let the rest fall through.
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<IREE::LinalgExt::IREELinalgExtDialect>();
    target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateMHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  mhlo::populateHloToLinalgConversionPattern(context, typeConverter, &patterns);
  // TODO(#5809): Drop ConcatenateOp lowering in favor of the upstream version
  //              then remove the PatternBenefit here
  patterns.insert<ScatterUpdateConversion>(typeConverter, context);
  patterns.insert<ConcatenateOpConversion, FftOpConversion>(
      typeConverter, context, PatternBenefit(1000));
}

std::unique_ptr<OperationPass<func::FuncOp>> createMHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertMHLOToLinalgOnTensorsPass>();
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
