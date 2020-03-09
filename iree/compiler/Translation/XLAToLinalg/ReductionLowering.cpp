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

//===- ReductionLowering.cpp ------------------------------------*-C++//-*-===//
//
// Lower reduction dispatch regions to Linalg.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
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

/// Checks whether an op is wthin an xla-hlo reduce region. During conversion,
/// the body of the reduce gets moved into a linalg.indexed_generic op. So check
/// if the op is within a linalg.indexed_generic op.
static bool isWithinReduceOpRegion(Operation *op) {
  return isa<linalg::IndexedGenericOp>(op->getParentOp());
}

namespace {

/// Pass to lower the reduction dispatch functions to Linalg.
struct HLOReductionToLinalgPass
    : public FunctionPass<HLOReductionToLinalgPass> {
  void runOnFunction() override;
};

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the `reductionDim`-th element.
// TODO(hanchung): Use helpers in StructuredOpsUtils.h instead of hardcoded
// strings once the build system is set up.
ArrayAttr getInnermostReductionIterAttrs(Builder b, unsigned nLoops) {
  SmallVector<Attribute, 3> attrs(nLoops, b.getStringAttr("parallel"));
  attrs.back() = b.getStringAttr("reduction");
  return b.getArrayAttr(attrs);
}

/// Base class for legalization of operations within the reduction apply
/// function (and the function itself).
template <typename OpTy>
class ReduceConversion : public OpConversionPattern<OpTy> {
 public:
  ReduceConversion(MLIRContext *context, TypeConverter &converter,
                   PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(context, benefit), converter(converter) {}

 protected:
  TypeConverter &converter;
};

/// Converts the xla_hlo.reduce op on tensors to a linalg.indexed_generic op on
/// buffers. Expects that the reduce op is the only op within the dispatch
/// function. This pattern also fuses std.constant operations which are defining
/// ops of the init value with the linalg.indexed_generic op.
struct ReduceOpConversion : public ReduceConversion<xla_hlo::ReduceOp> {
  using ReduceConversion<xla_hlo::ReduceOp>::ReduceConversion;
  PatternMatchResult matchAndRewrite(
      xla_hlo::ReduceOp reduceOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

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

//===----------------------------------------------------------------------===//
// Reduction Region op conversion.
//===----------------------------------------------------------------------===//

/// Operations within the reduction op region need to be converted to standard
/// ops.
template <typename OpTy>
struct ReduceRegionOpConversion final : public ReduceConversion<OpTy> {
  using ReduceConversion<OpTy>::ReduceConversion;

  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only convert it if it is within a reduce op region.
    if (!isWithinReduceOpRegion(op)) return this->matchFailure();
    if (operands.size() != 2) return this->matchFailure();
    SmallVector<Type, 1> resultElemTypes = {operands[0].getType()};
    Value opResult = xla_lhlo::MapXlaOpToStdScalarOp<OpTy>(op, resultElemTypes,
                                                           operands, &rewriter);
    rewriter.replaceOp(op, opResult);
    return this->matchSuccess();
  }
};

/// Converts xla_hlo.return to linalg.yield.
struct ReduceRegionReturnOpConversion final
    : public ReduceConversion<xla_hlo::ReturnOp> {
  using ReduceConversion<xla_hlo::ReturnOp>::ReduceConversion;
  PatternMatchResult matchAndRewrite(
      xla_hlo::ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!isWithinReduceOpRegion(op)) return this->matchFailure();
    if (operands.size() != 1) return matchFailure();
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, operands[0]);
    return matchSuccess();
  }
};

}  // namespace

/// Returns a permutation AffineMap that puts `reductionDim` to the last. The
/// order of the first (`rank` - 1) can be unsorted. E.g., if `rank` is 4 and
/// `reductionDim` is 1, then "(d0, d1, d2, d3) -> (d0, d3, d2, d1)" can be
/// returned.
static AffineMap getTransposeMapForReduction(OpBuilder &builder, int rank,
                                             int reductionDim) {
  SmallVector<unsigned, 4> permutation;
  for (int i = 0; i < rank; ++i) permutation.push_back(i);
  std::swap(permutation[reductionDim], permutation[rank - 1]);
  return AffineMap::getPermutationMap(permutation, builder.getContext());
}

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

PatternMatchResult ReduceOpConversion::matchAndRewrite(
    xla_hlo::ReduceOp reduceOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (reduceOp.getNumOperands() != 2) return matchFailure();
  Value src = *reduceOp.operands().begin();
  Value initVal = *reduceOp.init_values().begin();
  if (reduceOp.getNumResults() != 1) return matchFailure();

  // The result should be used to write into a memref using iree.store_output.
  Value result = reduceOp.getResult(0);
  if (!result.hasOneUse()) return matchFailure();
  auto storeOp = dyn_cast<IREE::StoreOutputOp>(result.use_begin()->getOwner());
  if (!storeOp) return matchFailure();
  Value dst = storeOp.dst();

  auto srcArgType = src.getType().template cast<ShapedType>();
  unsigned nInputRank = srcArgType.getRank();
  if (!nInputRank) return matchFailure();

  // Get the reduction dimension. For now expects only a single reduction
  // dimension.
  auto loc = reduceOp.getLoc();
  DenseIntElementsAttr dimensionsAttr = reduceOp.dimensions();
  Type attrType = dimensionsAttr.getType();
  if (attrType.cast<RankedTensorType>().getNumElements() != 1)
    return matchFailure();
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
    indexingMaps.emplace_back(AffineMapAttr::get(AffineMap::get(
        nInputRank, /*symbolCount=*/0, {rewriter.getAffineConstantExpr(0)})));
  // Since the reduction loop now is the innermost, the indexing map of `dst`
  // should drop the latest dimension, e.g., (d0, d1, d2) -> (d0, d1).
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0; i < nInputRank - 1; ++i)
    exprs.push_back(rewriter.getAffineDimExpr(i));
  if (exprs.empty()) exprs.push_back(rewriter.getAffineConstantExpr(0));
  indexingMaps.emplace_back(
      AffineMapAttr::get(AffineMap::get(nInputRank, /*symbolCount=*/0, exprs)));

  SmallVector<Type, 2> resultTypes = {};
  SmallVector<Value, 2> linalgOpArgs = {src};
  if (!initConstVal.hasValue()) linalgOpArgs.push_back(initVal);
  linalgOpArgs.push_back(dst);
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
    TypeConverter::SignatureConversion signatureConverter(linalgOpArgs.size());
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
  rewriter.replaceOp(reduceOp, dst);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// Pass for invoking the conversion
//===----------------------------------------------------------------------===//

void HLOReductionToLinalgPass::runOnFunction() {
  MLIRContext *context = &getContext();
  auto function = getFunction();

  // Only run this pass for dispatch regions with single xla_hlo::ReduceOp.
  // TODO(ravishankarm) : Move the above patterns into a pass where many such
  // conversion exist.
  if (!mlir::has_single_element(function.getBlocks()) ||
      !mlir::has_single_element(function.front().getOps<xla_hlo::ReduceOp>()))
    return;

  TypeConverter converter;
  converter.addConversion([](Type type) {
    return type.isSignlessIntOrFloat() ? type : Optional<Type>();
  });
  converter.addConversion([](RankedTensorType type) {
    return type.getRank() == 0 ? type.getElementType() : Optional<Type>();
  });

  OwningRewritePatternList patterns;
  patterns.insert<IREELoadInputOpConversion, IREEStoreOutputOpConversion>(
      context);
  patterns.insert<ReduceOpConversion, ReduceRegionOpConversion<xla_hlo::AddOp>,
                  ReduceRegionOpConversion<xla_hlo::MinOp>,
                  ReduceRegionOpConversion<xla_hlo::MaxOp>,
                  ReduceRegionReturnOpConversion>(context, converter);
  ConversionTarget target(*context);
  target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
  target.addLegalOp<FuncOp>();
  if (failed(applyFullConversion(function, target, patterns)))
    return signalPassFailure();
}

static PassRegistration<HLOReductionToLinalgPass> pass(
    "iree-hlo-reduction-to-linalg",
    "Convert the dispatch functions containing reduce operation to Linalg");

std::unique_ptr<Pass> createHLOReductionToLinalgPass() {
  return std::make_unique<HLOReductionToLinalgPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
