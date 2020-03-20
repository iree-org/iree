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

#include <cstddef>

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
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
  LogicalResult matchAndRewrite(
      IREE::LoadInputOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.src());
    return success();
  }
};

struct IREEStoreOutputOpConversion final
    : public OpConversionPattern<IREE::StoreOutputOp> {
  using OpConversionPattern<IREE::StoreOutputOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::StoreOutputOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (operands[0] != op.dst()) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// Returns a buffer to use for the `result` value of the operation. For now
/// checks if the result has a single use as the src operand of an
/// iree.store_output op. This can be generalized to allocate temp buffers
/// later.
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

  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 2> resultBuffers;
    resultBuffers.reserve(op.getOperation()->getNumResults());
    for (auto result : op.getOperation()->getResults()) {
      Value resultBuffer = getBufferForResult(result, rewriter);
      if (!resultBuffer) return failure();
      resultBuffers.push_back(resultBuffer);
    }

    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    LinalgOpTy linalgOp = static_cast<const DerivedTy &>(*this).apply(
        op, operands, resultBuffers, rewriter);

    if (!linalgOp || !linalgOp.hasBufferSemantics()) return failure();

    rewriter.replaceOp(op, linalgOp.getOutputBuffers());
    return success();
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
  if (op.dimension_numbers()) {
    const auto dimensionNumbers = op.dimension_numbers();
    const int inputSpatialRank =
        std::distance(dimensionNumbers.input_spatial_dimensions().begin(),
                      dimensionNumbers.input_spatial_dimensions().end());
    // Input storage order is N,spatial_dims...,Ci.
    if (dimensionNumbers.input_batch_dimension().getInt() != 0 ||
        dimensionNumbers.input_feature_dimension().getInt() !=
            (inputSpatialRank + 1))
      return nullptr;

    const int kernelSpatialRank =
        std::distance(dimensionNumbers.kernel_spatial_dimensions().begin(),
                      dimensionNumbers.kernel_spatial_dimensions().end());
    // Filter storage order is spatial_dims...,C, Co.
    if (dimensionNumbers.kernel_input_feature_dimension().getInt() !=
            kernelSpatialRank ||
        dimensionNumbers.kernel_output_feature_dimension().getInt() !=
            (kernelSpatialRank + 1))
      return nullptr;

    const int outputSpatialRank =
        std::distance(dimensionNumbers.output_spatial_dimensions().begin(),
                      dimensionNumbers.output_spatial_dimensions().end());
    // Output storage order is N,spatial_dims..,Co.
    if (dimensionNumbers.output_batch_dimension().getInt() != 0 ||
        dimensionNumbers.output_feature_dimension().getInt() !=
            (outputSpatialRank + 1))
      return nullptr;

    if (inputSpatialRank != outputSpatialRank ||
        inputSpatialRank != kernelSpatialRank)
      return nullptr;

    auto inputSpatialDim = dimensionNumbers.input_spatial_dimensions().begin();
    auto kernelSpatialDim =
        dimensionNumbers.kernel_spatial_dimensions().begin();
    auto outputSpatialDim =
        dimensionNumbers.output_spatial_dimensions().begin();
    // Check spatial dims are ordred correctly.
    for (int i = 0; i < inputSpatialRank; ++i) {
      const int dim = i + 1;
      if ((*inputSpatialDim++).getZExtValue() != dim ||
          (*outputSpatialDim++).getZExtValue() != dim ||
          (*kernelSpatialDim++).getZExtValue() != i)
        return nullptr;
    }
  }

  llvm::SmallVector<Attribute, 4> strides;
  llvm::SmallVector<Attribute, 4> dilation;
  if (op.window_strides().hasValue()) {
    auto range = op.window_strides().getValue().getAttributeValues();
    strides.append(range.begin(), range.end());
  }

  // TODO(ataei): Support dilated convolution only for now we need to add lhs
  // for deconvolution support
  if (op.rhs_dilation().hasValue()) {
    auto range = op.rhs_dilation().getValue().getAttributeValues();
    dilation.append(range.begin(), range.end());
  }

  auto stridesArg = ArrayAttr::get(strides, op.getContext());
  auto dilationArg = ArrayAttr::get(dilation, op.getContext());

  return rewriter.create<linalg::ConvOp>(op.getLoc(), args[1], args[0],
                                         results[0], stridesArg, dilationArg,
                                         op.paddingAttr());
}

// ----------------------------------------------------------------------------
// xla_hlo.reshape conversion patterns and utility functions.
// ----------------------------------------------------------------------------

namespace {
// Convert xla_hlo.reshape operation to linalg.copy op
// This reshape conversion supports only expanding or collapsing a single
// dimension.
// TODO(ataei): This is a workaround having a single ReshapeOp in a dispatch
// reigon. We should remove this once we have a linalg.ReshapeOp that can be
// tiled and fused with other ops.
struct ReshapeOpConversion
    : public ConvertToLinalgBufferOp<ReshapeOpConversion, xla_hlo::ReshapeOp,
                                     linalg::CopyOp> {
  using ConvertToLinalgBufferOp<ReshapeOpConversion, xla_hlo::ReshapeOp,
                                linalg::CopyOp>::ConvertToLinalgBufferOp;
  linalg::CopyOp apply(xla_hlo::ReshapeOp op, ArrayRef<Value> args,
                       ArrayRef<Value> results,
                       ConversionPatternRewriter &rewriter) const;
};
}  // namespace

// Finds a range [first, second) in dstShape where size(dstShape[first:second])
// = srcShape[first]. eg srcShape = [2, 16, 3], dstShape = [1, 2, 2, 4, 3] will
// return {1, 4}. If can't find such range it returns {-1, -1}.
static std::pair<int64_t, int64_t> findRange(ArrayRef<int64_t> srcShape,
                                             ArrayRef<int64_t> dstShape) {
  const std::pair<int64_t, int64_t> invalidRange = {-1, -1};
  int64_t start = 0, end = 0;
  int64_t srcRank = srcShape.size();
  int64_t dstRank = dstShape.size();
  while (start < srcRank && srcShape[start] == dstShape[start]) start++;
  if (start >= srcRank) return invalidRange;
  int64_t size = srcShape[start];
  int64_t dstSize = 1;
  end = start;
  while (end < dstRank) {
    dstSize *= dstShape[end];
    if (dstSize > size) break;
    end++;
  }
  if (end > dstRank) return invalidRange;
  // Check all remaining dims are equal.
  int ss = start + 1;
  int ee = end;
  while (ss < srcRank && ee < dstRank) {
    if (srcShape[ss++] != dstShape[ee++]) return invalidRange;
  }
  return {start, end};
}

linalg::CopyOp ReshapeOpConversion::apply(
    xla_hlo::ReshapeOp op, ArrayRef<Value> args, ArrayRef<Value> results,
    ConversionPatternRewriter &rewriter) const {
  // Reassociate dims from
  auto inShape = args[0].getType().cast<ShapedType>();
  auto outShape = results[0].getType().cast<ShapedType>();

  bool isCollapseDims = inShape.getRank() > outShape.getRank();

  auto range = isCollapseDims
                   ? findRange(outShape.getShape(), inShape.getShape())
                   : findRange(inShape.getShape(), outShape.getShape());

  if (range.first == -1 || range.second == -1) return nullptr;

  llvm::SmallVector<llvm::SmallVector<AffineExpr, 4>, 4> exprs(
      std::min(outShape.getRank(), inShape.getRank()));

  SmallVector<ArrayRef<AffineExpr>, 4> reassociationMaps;

  int dim = 0;
  for (int i = 0; i < std::max(inShape.getRank(), outShape.getRank()); ++i) {
    if (i >= range.first && i < range.second) {
      for (int j = range.first; j < range.second; ++j) {
        exprs[dim].push_back(rewriter.getAffineDimExpr(j));
      }
      i = range.second - 1;
      dim++;
    } else {
      exprs[dim++].push_back(rewriter.getAffineDimExpr(i));
    }
  }
  for (auto &expr : exprs) reassociationMaps.push_back(expr);

  linalg::ReshapeOp reshapeOp = rewriter.create<linalg::ReshapeOp>(
      op.getLoc(), results[0].getType(), args[0], reassociationMaps);

  return rewriter.create<linalg::CopyOp>(op.getLoc(), reshapeOp.getResult(),
                                         results[0]);
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
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
// TODO(hanchung): Use helpers in StructuredOpsUtils.h instead of hardcoded
// strings once the build system is set up.
static ArrayAttr getParallelAndReductionIterAttrs(Builder b, unsigned nLoops,
                                                  unsigned nReduction) {
  SmallVector<Attribute, 3> attrs(nLoops - nReduction,
                                  b.getStringAttr("parallel"));
  attrs.append(nReduction, b.getStringAttr("reduction"));
  return b.getArrayAttr(attrs);
}

/// Returns a permutation AffineMap that puts all reduction dimensions to the
/// last. The order of parallel loops and reduction loops are all sorted. E.g.,
/// if `rank` is 4 and `reductionDims` is {1, 3}, then
/// "(d0, d1, d2, d3) -> (d0, d2, d1, d3)" is used. The inverse permutation of
/// the AffineMap is returned.
static AffineMap getTransposeMapForReduction(MLIRContext *context, int rank,
                                             ArrayRef<int> reductionDims) {
  llvm::SmallSetVector<int, 4> s;
  for (auto dim : reductionDims) s.insert(dim);

  SmallVector<unsigned, 4> permutation;
  for (int i = 0; i < rank; ++i)
    if (!s.count(i)) permutation.push_back(i);
  for (auto dim : reductionDims) permutation.push_back(dim);

  auto map = AffineMap::getPermutationMap(permutation, context);
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
  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only convert it if it is within a reduce op region.
    if (!isWithinReduceOpRegion(op)) return failure();
    Operation *replacement =
        static_cast<const DerivedTy &>(*this).apply(op, operands, rewriter);
    if (!replacement) return failure();
    rewriter.replaceOp(op, replacement->getResults());
    return success();
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
  SmallVector<int, 4> reductionDims;
  for (auto dim : dimensionsAttr.getIntValues())
    reductionDims.push_back(dim.getSExtValue());

  // Check if initVal is constant. If so, inline the value into the region.
  Optional<Attribute> initConstVal = getInitValueAsConst(initVal);
  if (initConstVal.hasValue()) {
    if (initVal.hasOneUse()) rewriter.eraseOp(initVal.getDefiningOp());
    initVal = rewriter.create<ConstantOp>(initVal.getDefiningOp()->getLoc(),
                                          initConstVal.getValue());
  }

  // Prepare indexing maps for linalg generic op. The elements are for src,
  // initial value and dst, respectively.
  // Transpose `src` to make the reduction loops be the innermost, because it's
  // easier to fully utilize processors.
  SmallVector<Attribute, 3> indexingMaps;
  indexingMaps.emplace_back(AffineMapAttr::get(getTransposeMapForReduction(
      rewriter.getContext(), nInputRank, reductionDims)));
  if (!initConstVal.hasValue())
    indexingMaps.emplace_back(AffineMapAttr::get(
        AffineMap::get(nInputRank, /*symbolCount=*/0, rewriter.getContext())));
  // The indexing map of `dst` should drop the reduction loops. Since the
  // reduction loops now are all in the innermost, drops `reductionDims.size()`
  // dimensions. We don't need an inverse permutation here because they are the
  // same.
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0, e = nInputRank - reductionDims.size(); i < e; ++i)
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
      getParallelAndReductionIterAttrs(rewriter, nInputRank,
                                       reductionDims.size()),
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
    // The reduction dimensions are the innermost loops now, compare all
    // reduction indices to zero. If they are all zero, it's the first time to
    // update the output element, i.e., we should take initial value to compute
    // with the input element.
    Value zero = rewriter.create<ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, 0));
    Value cond = rewriter.create<ConstantOp>(loc, rewriter.getBoolAttr(true));
    for (int i = nInputRank - reductionDims.size(); i < nInputRank; ++i) {
      Value isZero = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq,
                                             entryBlock->getArgument(i), zero);
      cond = rewriter.create<AndOp>(loc, cond, isZero);
    }
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

void populateHLOToLinalgOnConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<ConvOpConversion, DotOpConversion, ReshapeOpConversion,
                  IREELoadInputOpConversion, IREEStoreOutputOpConversion,
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
  populateHLOToLinalgOnConversionPatterns(context, patterns);
  ConversionTarget target(*context);
  target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
  target.addDynamicallyLegalOp<linalg::GenericOp, linalg::IndexedGenericOp>(
      [](linalg::LinalgOp op) { return op.hasBufferSemantics(); });
  target.addLegalOp<FuncOp>();
  if (failed(applyFullConversion(getFunction(), target, patterns)))
    return signalPassFailure();
}

std::unique_ptr<OpPassBase<FuncOp>> createHLOToLinalgOnBuffersPass() {
  return std::make_unique<XLAToLinalgOnBuffersPass>();
}

static PassRegistration<XLAToLinalgOnBuffersPass> pass(
    "iree-hlo-to-linalg-on-buffers",
    "Convert from XLA-HLO ops to Linalg ops on buffers");
}  // namespace iree_compiler
}  // namespace mlir
