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

//===- XLAToLinalgOnBuffers.cpp - Pass to convert XLA to Linalg on buffers-===//
//
// Pass to convert from XLA to linalg on buffers. Currently only handles cases
// where the dispatch region contains a single xla_hlo op that can be converted
// to linalg on buffers.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// IREE dialect op conversions.
//===----------------------------------------------------------------------===//

// TODO(ravishankarm): These conversion patterns are there in a few
// places. Combine all the patterns into a single pass.
struct IREELoadInputOpConversion final
    : public OpConversionPattern<IREE::LoadInputOp> {
  using OpConversionPattern<IREE::LoadInputOp>::OpConversionPattern;
  PatternMatchResult matchAndRewrite(
      IREE::LoadInputOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.src());
    return matchSuccess();
  }
};

struct IREEStoreOutputOpConversion final
    : public OpConversionPattern<IREE::StoreOutputOp> {
  using OpConversionPattern<IREE::StoreOutputOp>::OpConversionPattern;
  PatternMatchResult matchAndRewrite(
      IREE::StoreOutputOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (operands[0] != op.dst()) return matchFailure();
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

/// Returns a buffer to use for the result of the operation. For now checks if
/// the results has a single use as the src operand of an iree.store_output
/// op. This can be generalized to allocate temp buffers later.
static Value getBufferForResult(Value result, OpBuilder &builder) {
  if (!result.hasOneUse()) return nullptr;
  auto resultMemrefOp =
      dyn_cast<IREE::StoreOutputOp>(result.use_begin()->getOwner());
  if (!resultMemrefOp) return nullptr;
  return resultMemrefOp.dst();
}

namespace {

/// Base class to convert to linalg on buffers. All derived classes implement an
/// apply method with the following signature
///
/// OpTy apply(SrcOpTy op, ArrayRef<Value> args, ArrayRef<Value> results,
///            ConversionPatternRewriter& rewriter) const;
///
/// The `op` is the op being converted. `args` contains the buffers to use for
/// as inputs to the converted op, and `results` contains the buffer to use for
/// the outputs of the converted op. The method returns a linalg op on buffers
template <typename DerivedTy, typename SrcOpTy, typename LinalgOpTy>
struct ConvertToLinalgBufferOp : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 2> resultBuffers;
    resultBuffers.reserve(op.getOperation()->getNumResults());
    for (auto result : op.getOperation()->getResults()) {
      Value resultBuffer = getBufferForResult(result, rewriter);
      if (!resultBuffer) return ConversionPattern::matchFailure();
      resultBuffers.push_back(resultBuffer);
    }

    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    LinalgOpTy linalgOp = static_cast<const DerivedTy &>(*this).apply(
        op, operands, resultBuffers, rewriter);
    if (!linalgOp || !linalgOp.hasBufferSemantics())
      return ConversionPattern::matchFailure();

    rewriter.replaceOp(op, linalgOp.getOutputBuffers());
    return ConversionPattern::matchSuccess();
  }
};

/// Converts xla_hlo.dot operation to linalg.matmul op
struct DotOpConversion
    : public ConvertToLinalgBufferOp<DotOpConversion, xla_hlo::DotOp,
                                     linalg::MatmulOp> {
  using ConvertToLinalgBufferOp<DotOpConversion, xla_hlo::DotOp,
                                linalg::MatmulOp>::ConvertToLinalgBufferOp;
  linalg::MatmulOp apply(xla_hlo::DotOp op, ArrayRef<Value> args,
                         ArrayRef<Value> results,
                         ConversionPatternRewriter &rewriter) const {
    return rewriter.create<linalg::MatmulOp>(op.getLoc(), args[0], args[1],
                                             results[0]);
  }
};
}  // namespace

// -----------------------------------------------------------------------------
// xla_hlo.conv conversion patterns and utility functions.
// -----------------------------------------------------------------------------

namespace {
/// Converts xla_hlo.conv operation linalg.conv op
struct ConvOpConversion
    : public ConvertToLinalgBufferOp<ConvOpConversion, xla_hlo::ConvOp,
                                     linalg::ConvOp> {
  using ConvertToLinalgBufferOp<ConvOpConversion, xla_hlo::ConvOp,
                                linalg::ConvOp>::ConvertToLinalgBufferOp;
  linalg::ConvOp apply(xla_hlo::ConvOp op, ArrayRef<Value> args,
                       ArrayRef<Value> results,
                       ConversionPatternRewriter &rewriter) const;
};
}  // namespace

linalg::ConvOp ConvOpConversion::apply(
    xla_hlo::ConvOp op, ArrayRef<Value> args, ArrayRef<Value> results,
    ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<Attribute, 4> strides;
  llvm::SmallVector<Attribute, 4> dilation;
  if (op.window_strides().hasValue()) {
    strides.insert(strides.begin(),
                   op.window_strides().getValue().getAttributeValues().begin(),
                   op.window_strides().getValue().getAttributeValues().end());
  }

  // TODO(ataei): Support dilated convolution only for now we need to add lhs
  // for deconvolution support
  if (op.rhs_dilation().hasValue()) {
    dilation.insert(dilation.begin(),
                    op.rhs_dilation().getValue().getAttributeValues().begin(),
                    op.rhs_dilation().getValue().getAttributeValues().end());
  }

  auto stridesArg = ArrayAttr::get(strides, op.getContext());
  auto dilationArg = ArrayAttr::get(dilation, op.getContext());

  return rewriter.create<linalg::ConvOp>(op.getLoc(), args[1], args[0],
                                         results[0], stridesArg, dilationArg);
}

// -----------------------------------------------------------------------------
// xla_hlo.reduce conversion patterns and utility functions.
// -----------------------------------------------------------------------------

/// Returns the constant value associated with the init value if the defining
/// operation is a constant.
static Optional<Attribute> getInitValueAsConst(Value init) {
  DenseElementsAttr attr;
  if (!matchPattern(init, m_Constant(&attr))) return {};
  auto type = attr.getType().dyn_cast<ShapedType>();
  if (!type || type.getRank() != 0) return {};
  if (auto intType = type.getElementType().dyn_cast<IntegerType>())
    return IntegerAttr::get(intType, attr.getValue<APInt>({}));
  else if (auto floatType = type.getElementType().dyn_cast<FloatType>())
    return FloatAttr::get(floatType, attr.getValue<APFloat>({}));
  return {};
}

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the `reductionDim`-th element.
// TODO(hanchung): Use helpers in StructuredOpsUtils.h instead of hardcoded
// strings once the build system is set up.
static ArrayAttr getInnermostReductionIterAttrs(Builder b, unsigned nLoops) {
  SmallVector<Attribute, 3> attrs(nLoops, b.getStringAttr("parallel"));
  attrs.back() = b.getStringAttr("reduction");
  return b.getArrayAttr(attrs);
}

/// Returns a permutation AffineMap that puts `reductionDim` to the last. The
/// order of the first (`rank` - 1) is sorted. E.g., if `rank` is 4 and
/// `reductionDim` is 1, then "(d0, d1, d2, d3) -> (d0, d2, d3, d1)" is used.
/// The inverse permutation of the AffineMap is returned.
static AffineMap getTransposeMapForReduction(OpBuilder &builder, int rank,
                                             int reductionDim) {
  SmallVector<unsigned, 4> permutation;
  for (int i = 0; i < rank; ++i) {
    if (i != reductionDim) permutation.push_back(i);
  }
  permutation.push_back(reductionDim);
  auto map = AffineMap::getPermutationMap(permutation, builder.getContext());
  return inversePermutation(map);
}

/// Checks whether an op is wthin an xla-hlo reduce region. During conversion,
/// the body of the reduce gets moved into a linalg.indexed_generic op. So check
/// if the op is within a linalg.indexed_generic op.
static bool isWithinReduceOpRegion(Operation *op) {
  return isa<linalg::IndexedGenericOp>(op->getParentOp());
}

namespace {

/// Type converter for converting the region of an xla_hlo::reduce op.
class ReduceRegionTypeConverter : public TypeConverter {
 public:
  Type convertType(Type type) const {
    if (type.isSignlessIntOrFloat()) {
      return type;
    } else if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
      if (tensorType.getRank() == 0) return tensorType.getElementType();
    }
    return nullptr;
  }
};

/// Converts the xla_hlo.reduce op on tensors to a linalg.indexed_generic op on
/// buffers. Expects that the reduce op is the only op within the dispatch
/// function. This pattern also fuses std.constant operations which are defining
/// ops of the init value with the linalg.indexed_generic op.
struct ReduceOpConversion
    : public ConvertToLinalgBufferOp<ReduceOpConversion, xla_hlo::ReduceOp,
                                     linalg::IndexedGenericOp> {
  using ConvertToLinalgBufferOp<
      ReduceOpConversion, xla_hlo::ReduceOp,
      linalg::IndexedGenericOp>::ConvertToLinalgBufferOp;
  linalg::IndexedGenericOp apply(xla_hlo::ReduceOp reduceOp,
                                 ArrayRef<Value> operands,
                                 ArrayRef<Value> results,
                                 ConversionPatternRewriter &rewriter) const;

 private:
  ReduceRegionTypeConverter converter;
};

/// Base class for converting operations within the reduction op region. Derived
/// classes implement the following apply method to implement the conversion.
///    Value apply(OpTy op, ArrayRef<Value> args, ConversionPatternRewriter
///    &rewriter) const;
template <typename DerivedTy, typename OpTy>
struct ReduceRegionOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only convert it if it is within a reduce op region.
    if (!isWithinReduceOpRegion(op)) return this->matchFailure();
    Operation *replacement =
        static_cast<const DerivedTy &>(*this).apply(op, operands, rewriter);
    if (!replacement) return this->matchFailure();
    rewriter.replaceOp(op, replacement->getResults());
    return this->matchSuccess();
  }

 protected:
  ReduceRegionTypeConverter converter;
};

/// Converts XLA ops within reduce region to standard ops.
template <typename OpTy>
struct ReduceRegionXLAOpConversion final
    : public ReduceRegionOpConversion<ReduceRegionXLAOpConversion<OpTy>, OpTy> {
  using ReduceRegionOpConversion<ReduceRegionXLAOpConversion<OpTy>,
                                 OpTy>::ReduceRegionOpConversion;
  Operation *apply(OpTy op, ArrayRef<Value> operands,
                   ConversionPatternRewriter &rewriter) const {
    Value result = xla_lhlo::XlaOpToStdScalarOp::map<OpTy>(
        op, operands[0].getType(), operands, &rewriter);
    return result.getDefiningOp();
  }
};

/// Converts xla_hlo.return to within a reduce region to a linalg.yield.
struct ReduceRegionReturnOpConversion final
    : public ReduceRegionOpConversion<ReduceRegionReturnOpConversion,
                                      xla_hlo::ReturnOp> {
  using ReduceRegionOpConversion<ReduceRegionReturnOpConversion,
                                 xla_hlo::ReturnOp>::ReduceRegionOpConversion;
  Operation *apply(xla_hlo::ReturnOp op, ArrayRef<Value> operands,
                   ConversionPatternRewriter &rewriter) const {
    return rewriter.create<linalg::YieldOp>(op.getLoc(), operands[0]);
  }
};
}  // namespace

linalg::IndexedGenericOp ReduceOpConversion::apply(
    xla_hlo::ReduceOp reduceOp, ArrayRef<Value> operands,
    ArrayRef<Value> results, ConversionPatternRewriter &rewriter) const {
  if (reduceOp.getNumOperands() != 2) return nullptr;
  Value src = *reduceOp.operands().begin();
  Value initVal = *reduceOp.init_values().begin();
  if (reduceOp.getNumResults() != 1) return nullptr;

  auto srcArgType = src.getType().template cast<ShapedType>();
  unsigned nInputRank = srcArgType.getRank();
  if (!nInputRank) return nullptr;

  // Get the reduction dimension. For now expects only a single reduction
  // dimension.
  auto loc = reduceOp.getLoc();
  DenseIntElementsAttr dimensionsAttr = reduceOp.dimensions();
  Type attrType = dimensionsAttr.getType();
  if (attrType.cast<RankedTensorType>().getNumElements() != 1) return nullptr;
  int reductionDim = dimensionsAttr.getValue<APInt>(0).getSExtValue();

  // Check if initVal is constant. If so, inline the value into the region.
  Optional<Attribute> initConstVal = getInitValueAsConst(initVal);
  if (initConstVal.hasValue()) {
    if (initVal.hasOneUse()) rewriter.eraseOp(initVal.getDefiningOp());
    initVal = rewriter.create<ConstantOp>(initVal.getDefiningOp()->getLoc(),
                                          initConstVal.getValue());
  }

  // Prepare indexing maps for linalg generic op. The elements are for src,
  // initial value and dst, respectively.
  // Transpose `src` to make the reduction loop be the innermost, because it's
  // easier to fully utilize processors.
  SmallVector<Attribute, 3> indexingMaps;
  indexingMaps.emplace_back(AffineMapAttr::get(
      getTransposeMapForReduction(rewriter, nInputRank, reductionDim)));
  if (!initConstVal.hasValue())
    indexingMaps.emplace_back(AffineMapAttr::get(
        AffineMap::get(nInputRank, /*symbolCount=*/0, rewriter.getContext())));
  // Since the reduction loop now is the innermost and the parallel loops are
  // sorted, the indexing map of `dst` should drop the latest dimension, e.g.,
  // (d0, d1, d2) -> (d0, d1). We don't need an inverse permutation here because
  // they are the same.
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0; i < nInputRank - 1; ++i)
    exprs.push_back(rewriter.getAffineDimExpr(i));
  indexingMaps.emplace_back(AffineMapAttr::get(
      exprs.empty()
          ? AffineMap::get(nInputRank, /*symbolCount=*/0, rewriter.getContext())
          : AffineMap::get(nInputRank, /*symbolCount=*/0, exprs)));

  SmallVector<Type, 2> resultTypes = {};
  SmallVector<Value, 2> linalgOpArgs = {operands[0]};
  if (!initConstVal.hasValue()) linalgOpArgs.push_back(operands[1]);
  linalgOpArgs.push_back(results[0]);
  auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
      loc, resultTypes, linalgOpArgs,
      rewriter.getI64IntegerAttr(linalgOpArgs.size() - 1),  // args_in
      rewriter.getI64IntegerAttr(1),                        // args_out
      rewriter.getArrayAttr(indexingMaps),
      getInnermostReductionIterAttrs(rewriter, nInputRank),
      /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

  linalgOp.region().takeBody(reduceOp.body());
  {
    OpBuilder::InsertionGuard regionGuard(rewriter);

    // Convert the signature of the body. The reduce op region apply function
    // has a signature (lhs, rhs) -> output, all of the same tensor type t. This
    // is converted to a function with the same signature but with element
    // types. E.g., "(tensor<f32>, tensor<f32>) -> tensor<f32>" will be
    // converted to "(f32, f32, f32)".
    TypeConverter::SignatureConversion signatureConverter(2);
    Type argType = linalgOp.region().front().getArgument(0).getType();
    Type convertedType = converter.convertType(argType);
    Type indexType = rewriter.getIndexType();
    for (unsigned i = 0; i < nInputRank; ++i) {
      signatureConverter.addInputs(indexType);
    }
    signatureConverter.addInputs(0, convertedType);
    if (!initConstVal.hasValue()) signatureConverter.addInputs(convertedType);
    signatureConverter.addInputs(1, convertedType);
    Block *entryBlock = rewriter.applySignatureConversion(&linalgOp.region(),
                                                          signatureConverter);

    // The indexed generic op generated here combines the input value with the
    // init value for the zero-th iteration of the reduction loop. This is
    // yielded by the region to model a store of the value to the output. The
    // input value with the output value for all other iterations.
    unsigned numArgs = entryBlock->getNumArguments();
    BlockArgument blockDstArg = entryBlock->getArgument(numArgs - 1);
    rewriter.setInsertionPointToStart(entryBlock);
    Value initArg = nullptr;
    if (initConstVal.hasValue())
      initArg = initVal;
    else
      initArg = entryBlock->getArgument(numArgs - 2);
    Value zero = rewriter.create<ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, 0));
    // The reduction dimension is the innermost loop now, so compare the
    // innermost index to zero.
    Value cond = rewriter.create<CmpIOp>(
        loc, CmpIPredicate::eq, entryBlock->getArgument(nInputRank - 1), zero);
    Value lhs = rewriter.create<SelectOp>(loc, cond, initArg, blockDstArg);
    rewriter.replaceUsesOfBlockArgument(blockDstArg, lhs);
  }
  return linalgOp;
}

// -----------------------------------------------------------------------------
// linalg op on tensors to linalg op on buffers.
// -----------------------------------------------------------------------------
namespace {
template <typename LinalgOpTy>
struct LinalgOpOnTensorConversion
    : public ConvertToLinalgBufferOp<LinalgOpOnTensorConversion<LinalgOpTy>,
                                     LinalgOpTy, LinalgOpTy> {
  using ConvertToLinalgBufferOp<LinalgOpOnTensorConversion<LinalgOpTy>,
                                LinalgOpTy,
                                LinalgOpTy>::ConvertToLinalgBufferOp;
  LinalgOpTy apply(LinalgOpTy op, ArrayRef<Value> args, ArrayRef<Value> results,
                   ConversionPatternRewriter &rewriter) const {
    if (!op.hasTensorSemantics()) return nullptr;
    SmallVector<Value, 2> opArgs(args.begin(), args.end());
    opArgs.append(results.begin(), results.end());

    // Create a new op with the same traits as the original generic op, but with
    // memrefs.
    // TODO(ravishankarm): Figure out how to do this inplace.
    auto linalgBufferOp = rewriter.template create<LinalgOpTy>(
        op.getLoc(), ArrayRef<Type>(), opArgs, op.args_in(), op.args_out(),
        op.indexing_maps(), op.iterator_types(),
        /*doc=*/nullptr,
        /*fun=*/nullptr,
        /*library_call=*/nullptr);
    // Move the region from the replaced op into the new op.
    unsigned numTensorOperands = op.getNumOperands();
    auto &region = linalgBufferOp.region();
    region.takeBody(op.region());
    // Need to convert the signature to take extra arguments for the return
    // type.
    TypeConverter::SignatureConversion signatureConverter(numTensorOperands);
    for (auto arg : llvm::enumerate(opArgs)) {
      if (arg.index() < numTensorOperands) {
        signatureConverter.addInputs(
            arg.index(),
            arg.value().getType().cast<MemRefType>().getElementType());
      } else {
        signatureConverter.addInputs(
            arg.value().getType().cast<MemRefType>().getElementType());
      }
    }
    rewriter.applySignatureConversion(&region, signatureConverter);
    return linalgBufferOp;
  }
};
}  // namespace

// -----------------------------------------------------------------------------
// Pass specification.
// -----------------------------------------------------------------------------

namespace {
struct XLAToLinalgOnBuffersPass
    : public FunctionPass<XLAToLinalgOnBuffersPass> {
  void runOnFunction() override;
};
}  // namespace

void populateXLAToLinalgOnConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<ConvOpConversion, DotOpConversion, IREELoadInputOpConversion,
                  IREEStoreOutputOpConversion,
                  LinalgOpOnTensorConversion<linalg::GenericOp>>(context);
  // Reduce Operation conversions.
  patterns
      .insert<ReduceOpConversion, ReduceRegionXLAOpConversion<xla_hlo::AddOp>,
              ReduceRegionXLAOpConversion<xla_hlo::MinOp>,
              ReduceRegionXLAOpConversion<xla_hlo::MaxOp>,
              ReduceRegionReturnOpConversion>(context);
}

void XLAToLinalgOnBuffersPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto *context = &getContext();
  populateXLAToLinalgOnConversionPatterns(context, patterns);
  ConversionTarget target(*context);
  target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
  target.addDynamicallyLegalOp<linalg::GenericOp, linalg::IndexedGenericOp>(
      [](linalg::LinalgOp op) { return op.hasBufferSemantics(); });
  target.addLegalOp<FuncOp>();
  if (failed(applyFullConversion(getFunction(), target, patterns)))
    return signalPassFailure();
}

std::unique_ptr<OpPassBase<FuncOp>> createXLAToLinalgOnBuffersPass() {
  return std::make_unique<XLAToLinalgOnBuffersPass>();
}

static PassRegistration<XLAToLinalgOnBuffersPass> pass(
    "iree-hlo-to-linalg-on-buffers",
    "Convert XLA-HLO ops to Linalg on Buffer ops");
}  // namespace iree_compiler
}  // namespace mlir
