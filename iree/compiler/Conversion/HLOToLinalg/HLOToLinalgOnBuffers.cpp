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

//===- HLOToLinalgOnBuffers.cpp - Pass to convert HLO to Linalg on buffers-===//
//
// Pass to convert from HLO to linalg on buffers. Currently only handles cases
// where the dispatch region contains a single xla_hlo op that can be converted
// to linalg on buffers.
//
//===----------------------------------------------------------------------===//

#include <cstddef>

#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"

namespace mlir {
namespace iree_compiler {

// -----------------------------------------------------------------------------
// Utility functions.
// -----------------------------------------------------------------------------

static std::vector<int64_t> convertDenseIntAttr(
    mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

/// Returns the constant value associated with the init value if the defining
/// operation is a constant.
static Attribute getInitValueAsConst(Value init) {
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

/// Emits linalg.fill op to fill the given `buffer` with zero value.
static LogicalResult zeroFillBuffer(Location loc, Value buffer,
                                    OpBuilder &builder) {
  auto zeroAttr =
      builder.getZeroAttr(buffer.getType().cast<MemRefType>().getElementType());
  if (!zeroAttr) return failure();
  auto zeroValue = builder.create<ConstantOp>(loc, zeroAttr);
  builder.create<linalg::FillOp>(loc, buffer, zeroValue);
  return success();
}

//===----------------------------------------------------------------------===//
// Linalg tensor and buffer conversion utilities.
//===----------------------------------------------------------------------===//

/// Returns the memory space for the given descriptor `type`.
// Note: This function should be kept in consistence with SPIRVTypeConverter's
// getMemorySpaceForStorageClass(). But it does not make sense to directly use
// that here.
static unsigned mapDescriptorTypeToMemorySpace(IREE::HAL::DescriptorType type) {
  switch (type) {
    case IREE::HAL::DescriptorType::StorageBuffer:
    case IREE::HAL::DescriptorType::StorageBufferDynamic:
      return 0;
    case IREE::HAL::DescriptorType::UniformBuffer:
    case IREE::HAL::DescriptorType::UniformBufferDynamic:
      return 4;
  }
}

/// Returns the MemRefType to use for a given `tensorType`.
static MemRefType getMemrefTypeForTensor(
    RankedTensorType tensorType, ArrayRef<AffineMap> affineMapComposition = {},
    unsigned memorySpace = 0) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         affineMapComposition, memorySpace);
}

/// Returns the MemRefType to use for a `value` of type RankedTensorType.
static MemRefType getMemrefTypeForTensor(
    Value value, ArrayRef<AffineMap> affineMapComposition = {},
    unsigned memorySpace = 0) {
  return getMemrefTypeForTensor(value.getType().cast<RankedTensorType>());
}

/// Returns a corresponding memref type for the given `tensorType` stored in the
/// given `descriptorType`.
static MemRefType getTensorBackingBufferType(
    RankedTensorType tensorType, IREE::HAL::DescriptorType descriptorType) {
  // Get the memory space from the HAL interface so we can carry that over via
  // memref.
  return getMemrefTypeForTensor(tensorType, /*affineMapComposition=*/{},
                                mapDescriptorTypeToMemorySpace(descriptorType));
}

/// Resolves the given `result` tensor to the corresponding buffer backing it if
/// the given `operand` buffer has been assigned a backing buffer and that
/// buffer is the same as `replacement`. Returns nullptr on failure.
///
/// This is based on the assumption that the view-like operation chain that
/// manipulates the tensors are processed in the reverse order when assigning
/// backing buffers to tensors, so if an operand tensor to a view-like op is
/// resolved, then the result buffer for it must also be resolved.
static Value resolveResult(Value operand, Value replacement, Value result,
                           TensorToBufferMap const &resultTensorToBufferMap) {
  return resultTensorToBufferMap.lookup(operand) == replacement
             ? resultTensorToBufferMap.lookup(result)
             : nullptr;
}

namespace {
//===----------------------------------------------------------------------===//
// Linalg on buffers conversion base class.
//===----------------------------------------------------------------------===//

/// Base class to convert linalg on tensors to Linalg on buffers.
///
/// This base class handles getting/allocating interface buffers for the Linalg
/// op inputs and outputs, so that all derived classes can assume the inputs and
/// outputs are already buffers and perform the main conversion logic.
//
/// All derived classes implement a static apply method with the following
/// signature:
///
/// ```c++
/// LogicalResult apply(SrcOpTy op, ArrayRef<Value> inputBuffers,
///                     ArrayRef<Value> resultBuffers,
///                     ConversionPatternRewriter& rewriter) const;
/// ```
///
/// The `op` is the op being converted. `inputBuffers` contains the buffers to
/// use for as inputs to the converted op, and `resultBuffers` contains the
/// buffer to use for the outputs of the converted op. The method returns a
/// linalg op on buffers.
template <typename DerivedTy, typename SrcOpTy>
struct ConvertToLinalgBufferOp : public OpConversionPattern<SrcOpTy> {
  ConvertToLinalgBufferOp(MLIRContext *context,
                          TensorToBufferMap const &resultTensorToBufferMap,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<SrcOpTy>(context, benefit),
        resultTensorToBufferMap(resultTensorToBufferMap) {}

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Operation *op = srcOp.getOperation();

    // Prepare interface buffers for results.
    SmallVector<Value, 1> resultBuffers;
    resultBuffers.reserve(op->getNumResults());
    for (auto result : llvm::enumerate(op->getResults())) {
      Value resultBuffer = resultTensorToBufferMap.lookup(result.value());
      if (!resultBuffer) {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag << "failed to create buffer for result #" << result.index();
        });
      }
      resultBuffers.push_back(resultBuffer);
    }

    // Apply the main conversion logic.
    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    if (failed(static_cast<DerivedTy const *>(this)->apply(
            srcOp, operands, resultBuffers, rewriter))) {
      return rewriter.notifyMatchFailure(
          op, "failed to apply main conversion logic");
    }

    // Ops using this Linalg op's results are expecting tensors. But here we
    // feed them buffers. This is okay because it is hidden as internal state
    // during conversion process. But this relies on collaborating patterns to
    // properly handle ops using the results.
    rewriter.replaceOp(srcOp, resultBuffers);
    return success();
  }

 protected:
  /// Map from tensor value that is a result of the dispatch function to the
  /// buffer that holds the result
  TensorToBufferMap const &resultTensorToBufferMap;
};
}  // namespace

//===----------------------------------------------------------------------===//
// xla_hlo.dot conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
enum class DotOperationType {
  VectorDot = 0,
  MatrixVector = 1,
  MatrixMatrix = 2,
  Unsupported = 3
};
}

static DotOperationType getDotOperationType(xla_hlo::DotOp dotOp) {
  ArrayRef<int64_t> lhsShape =
      dotOp.lhs().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape =
      dotOp.rhs().getType().cast<ShapedType>().getShape();
  auto shapeMatches = [](int64_t a, int64_t b) {
    return a == ShapedType::kDynamicSize || b == ShapedType::kDynamicSize ||
           a == b;
  };
  if (lhsShape.size() == 1 && rhsShape.size() == 1 &&
      shapeMatches(lhsShape[0], rhsShape[0]))
    return DotOperationType::VectorDot;
  if (lhsShape.size() == 2 && rhsShape.size() == 1 &&
      shapeMatches(lhsShape[1], rhsShape[0]))
    return DotOperationType::MatrixVector;
  if (rhsShape.size() == 2 && rhsShape.size() == 2 &&
      shapeMatches(lhsShape[1], rhsShape[0]))
    return DotOperationType::MatrixMatrix;
  return DotOperationType::Unsupported;
}

namespace {
/// Converts xla_hlo.dot operation to linalg.matmul op
template <DotOperationType opType, typename LinalgOpTy>
struct DotOpConversion
    : public ConvertToLinalgBufferOp<DotOpConversion<opType, LinalgOpTy>,
                                     xla_hlo::DotOp> {
  using ConvertToLinalgBufferOp<DotOpConversion<opType, LinalgOpTy>,
                                xla_hlo::DotOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(xla_hlo::DotOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const {
    if (getDotOperationType(op) == opType) {
      if (failed(zeroFillBuffer(op.getLoc(), resultBuffers[0], rewriter))) {
        rewriter.notifyMatchFailure(op, "failed to zero fill result buffer");
        return failure();
      }
      rewriter.create<LinalgOpTy>(op.getLoc(), inputBuffers[0], inputBuffers[1],
                                  resultBuffers[0]);
      return success();
    }
    return failure();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// xla_hlo.convolution conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Converts xla_hlo.convolution operation to linalg.conv op.
struct ConvOpConversion
    : public ConvertToLinalgBufferOp<ConvOpConversion, xla_hlo::ConvOp> {
  using ConvertToLinalgBufferOp<ConvOpConversion,
                                xla_hlo::ConvOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(xla_hlo::ConvOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;
};
}  // namespace

LogicalResult ConvOpConversion::apply(
    xla_hlo::ConvOp op, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  if (const auto dimensionNumbers = op.dimension_numbers()) {
    const int inputSpatialRank =
        llvm::size(dimensionNumbers.input_spatial_dimensions());
    // The dimensions for input should follow the order of
    // batch_count, spatial_dims..., input_feature_count.
    if (dimensionNumbers.input_batch_dimension().getInt() != 0 ||
        dimensionNumbers.input_feature_dimension().getInt() !=
            (inputSpatialRank + 1))
      return failure();

    const int kernelSpatialRank =
        llvm::size(dimensionNumbers.kernel_spatial_dimensions());
    // The dimensions for filter should follow the order of
    // spatial_dims..., input_feature_count, num_output_feature_count.
    if (dimensionNumbers.kernel_input_feature_dimension().getInt() !=
            kernelSpatialRank ||
        dimensionNumbers.kernel_output_feature_dimension().getInt() !=
            (kernelSpatialRank + 1))
      return failure();

    const int outputSpatialRank =
        llvm::size(dimensionNumbers.output_spatial_dimensions());
    // The dimensions for output should follow the order of
    // batch_count, spatial_dims.., output_feature_count.
    if (dimensionNumbers.output_batch_dimension().getInt() != 0 ||
        dimensionNumbers.output_feature_dimension().getInt() !=
            (outputSpatialRank + 1))
      return failure();

    if (inputSpatialRank != outputSpatialRank ||
        inputSpatialRank != kernelSpatialRank)
      return failure();

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
        return failure();
    }
  }

  llvm::SmallVector<Attribute, 4> strides;
  if (auto windowStrides = op.window_strides()) {
    auto range = windowStrides->getAttributeValues();
    strides.append(range.begin(), range.end());
  }
  auto stridesArg = ArrayAttr::get(strides, op.getContext());

  // TODO(ataei): Only support dilated convolution for now. We need to consider
  // LHS dilation for deconvolution cases.
  llvm::SmallVector<Attribute, 4> dilation;
  if (auto rhsDilation = op.rhs_dilation()) {
    auto range = rhsDilation->getAttributeValues();
    dilation.append(range.begin(), range.end());
  }
  auto dilationArg = ArrayAttr::get(dilation, op.getContext());

  // Set padding only if it is non-zero.
  DenseIntElementsAttr padding = op.paddingAttr();
  if (!padding || !llvm::any_of(padding.getValues<APInt>(), [](APInt intVal) {
        return !intVal.isNullValue();
      })) {
    padding = nullptr;
  }

  if (failed(zeroFillBuffer(op.getLoc(), resultBuffers[0], rewriter))) {
    rewriter.notifyMatchFailure(op, "failed to zero fill result buffer");
    return failure();
  }
  rewriter.create<linalg::ConvOp>(op.getLoc(), inputBuffers[1], inputBuffers[0],
                                  resultBuffers[0], stridesArg, dilationArg,
                                  padding);
  return success();
}

//===----------------------------------------------------------------------===//
// xla_hlo.pad conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Converts xla_hlo.pad operation to linalg.indexed_generic op.
// TODO(#1604): Lower the pad op to a Linalg named op.
struct PadOpConversion
    : public ConvertToLinalgBufferOp<PadOpConversion, xla_hlo::PadOp> {
  using ConvertToLinalgBufferOp<PadOpConversion,
                                xla_hlo::PadOp>::ConvertToLinalgBufferOp;

  LogicalResult apply(xla_hlo::PadOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;
};
}  // namespace

/// Returns an AffineMapAttr that is the indexing map to use for the input of a
/// xla_hlo.pad `op`.
static AffineMapAttr getPadOpInputIndexingMap(
    xla_hlo::PadOp op, int rank, ConversionPatternRewriter &rewriter) {
  const auto edgePaddingLow = convertDenseIntAttr(op.edge_padding_low());
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0; i < rank; ++i)
    exprs.push_back((rewriter.getAffineDimExpr(i) - edgePaddingLow[i]));
  return AffineMapAttr::get(
      AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext()));
}

LogicalResult PadOpConversion::apply(
    xla_hlo::PadOp op, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  if (llvm::any_of(op.interior_padding().getValues<IntegerAttr>(),
                   [](auto attr) { return attr.getInt() != 0; }))
    return op.emitError(
        "pad op with non-zero interiror_padding is not supported");

  xla_hlo::PadOp::Adaptor adaptor(inputBuffers);
  auto loc = op.getLoc();

  Attribute paddingConstVal = getInitValueAsConst(adaptor.padding_value());
  Value paddingVal =
      paddingConstVal
          ? rewriter.create<ConstantOp>(loc, paddingConstVal).getResult()
          : adaptor.padding_value();

  auto operandType = adaptor.operand().getType().cast<ShapedType>();
  int rank = operandType.getRank();

  SmallVector<Attribute, 2> indexingMaps;
  indexingMaps.emplace_back(getPadOpInputIndexingMap(op, rank, rewriter));
  if (!paddingConstVal) {
    indexingMaps.emplace_back(AffineMapAttr::get(
        AffineMap::get(rank, /*symbolCount=*/0, rewriter.getContext())));
  }
  indexingMaps.emplace_back(AffineMapAttr::get(
      AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())));

  SmallVector<Type, 2> resultTypes = {};
  SmallVector<Value, 2> linalgOpArgs = {adaptor.operand()};
  if (!paddingConstVal) linalgOpArgs.push_back(adaptor.padding_value());
  linalgOpArgs.push_back(resultBuffers[0]);
  auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
      loc, resultTypes, linalgOpArgs,
      rewriter.getI64IntegerAttr(linalgOpArgs.size() - 1),  // args_in
      rewriter.getI64IntegerAttr(1),                        // args_out
      rewriter.getArrayAttr(indexingMaps),
      getParallelAndReductionIterAttrs(rewriter, rank, /*nReduction=*/0),
      /*doc=*/nullptr, /*library_call=*/nullptr);

  // Add a block to the region.
  auto *region = &linalgOp.region();
  auto *block = rewriter.createBlock(region, region->end());
  SmallVector<Type, 4> bodyArgTypes;
  bodyArgTypes.append(rank, rewriter.getIndexType());
  bodyArgTypes.append(linalgOpArgs.size(), operandType.getElementType());
  block->addArguments(bodyArgTypes);
  rewriter.setInsertionPointToEnd(block);

  // If the `index` of the result at a particular dimension i, is d_i, check if
  //
  // (d_i >= edge_padding_low[i]) &&
  // (d_i < (edge_padding_low[i] + operand_shape[i])).
  //
  // If true, then use the value of the operand, otherwise use the padding
  // value.
  const auto &edgePaddingLow = op.edge_padding_low();
  const auto &edgePaddingHigh = op.edge_padding_high();

  Type indexType = rewriter.getIndexType();
  Value cond = nullptr;
  auto applyAndOp = [&](Value val) {
    cond = cond ? rewriter.create<AndOp>(loc, cond, val) : val;
  };
  for (int i = 0; i < rank; ++i) {
    Value dim = block->getArgument(i);
    int64_t paddingLow = edgePaddingLow.getValue<IntegerAttr>(i).getInt();
    int64_t paddingHigh = edgePaddingHigh.getValue<IntegerAttr>(i).getInt();
    auto low = rewriter.create<ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, paddingLow));

    // d_i < (edge_padding_low[i] + operand_shape[i])
    if (paddingLow != 0 && paddingHigh != 0) {
      auto operandExtent = rewriter.create<DimOp>(loc, adaptor.operand(), i);
      auto bound = rewriter.create<AddIOp>(loc, low, operandExtent);
      auto checkUb =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, dim, bound);
      applyAndOp(checkUb);
    }

    if (paddingLow != 0) {
      // d_i >= edge_padding_low[i]
      auto checkLb = rewriter.create<CmpIOp>(loc, CmpIPredicate::sge, dim, low);
      applyAndOp(checkLb);
    }
  }
  Value inputVal = block->getArgument(rank);
  if (!paddingConstVal) paddingVal = block->getArgument(rank + 1);
  Value result =
      cond ? rewriter.create<SelectOp>(loc, cond, inputVal, paddingVal)
           : inputVal;
  rewriter.create<linalg::YieldOp>(loc, result);

  setNoTileMarker(linalgOp);
  return success();
}

//===----------------------------------------------------------------------===//
// xla_hlo.torch_index_select conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts xla-hlo.torch_index_select op to a linalg.indexed_generic op.
/// Different from other ops on buffers, torch_index_select op needs indirect
/// access based on the `index` operand. Thus, an accessing on buffer is
/// involved inside the indexed_generic op and the input buffer is not passed as
/// an argument of the op. However, it doesn't affect anything on dependency
/// graph. It is just a magic buffer outside operations.
struct TorchIndexSelectOpConversion
    : public ConvertToLinalgBufferOp<TorchIndexSelectOpConversion,
                                     xla_hlo::TorchIndexSelectOp> {
  using ConvertToLinalgBufferOp<
      TorchIndexSelectOpConversion,
      xla_hlo::TorchIndexSelectOp>::ConvertToLinalgBufferOp;

  LogicalResult apply(xla_hlo::TorchIndexSelectOp op,
                      ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;
};
}  // namespace

LogicalResult TorchIndexSelectOpConversion::apply(
    xla_hlo::TorchIndexSelectOp op, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  xla_hlo::TorchIndexSelectOp::Adaptor adaptor(inputBuffers);
  int axis = op.dim().getSExtValue();
  int batch = op.batch_dims().getSExtValue();
  auto indexShapeType = adaptor.index().getType().dyn_cast<ShapedType>();
  int nIndices = indexShapeType.getRank();
  if (batch < 0)
    return op.emitError("expected batch_dims is greater than or equal to zero");

  Location loc = op.getLoc();
  Value output = op.getResult();
  int rank = output.getType().cast<ShapedType>().getRank();
  SmallVector<Attribute, 2> indexingMaps;
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0; i < batch; ++i) exprs.push_back(rewriter.getAffineDimExpr(i));
  for (int i = 0, e = nIndices - batch; i < e; ++i)
    exprs.push_back(rewriter.getAffineDimExpr(axis + i));
  indexingMaps.emplace_back(AffineMapAttr::get(
      AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext())));
  indexingMaps.emplace_back(
      AffineMapAttr::get(rewriter.getMultiDimIdentityMap(rank)));

  SmallVector<Type, 4> bodyArgTypes, opResultTypes;
  SmallVector<Value, 2> linalgOpArgs = {adaptor.index(), resultBuffers[0]};
  auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
      loc, opResultTypes, linalgOpArgs,
      rewriter.getI64IntegerAttr(1),  // args_in
      rewriter.getI64IntegerAttr(1),  // args_out
      rewriter.getArrayAttr(indexingMaps),
      getParallelAndReductionIterAttrs(rewriter, rank, /*nReduction=*/0),
      /*doc=*/nullptr, /*library_call=*/nullptr);

  // Add a block to the region.
  auto *region = &linalgOp.region();
  auto *block = rewriter.createBlock(region, region->end());
  bodyArgTypes.append(rank, rewriter.getIndexType());
  for (auto blockArgs : linalgOpArgs) {
    bodyArgTypes.push_back(
        blockArgs.getType().cast<ShapedType>().getElementType());
  }
  block->addArguments(bodyArgTypes);
  rewriter.setInsertionPointToEnd(block);

  SmallVector<Value, 4> indices;
  Value castedValue = rewriter.create<IndexCastOp>(
      loc, block->getArgument(rank), rewriter.getIndexType());
  for (int i = 0; i < axis; ++i) indices.push_back(block->getArgument(i));
  indices.push_back(castedValue);
  for (int i = axis + nIndices - batch; i < rank; ++i)
    indices.push_back(block->getArgument(i));

  Value res = rewriter.create<LoadOp>(loc, adaptor.input(), indices);
  rewriter.create<linalg::YieldOp>(loc, res);

  return success();
}

//===----------------------------------------------------------------------===//
// xla_hlo.reduce_window conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {

/// xla_hlo.reduce_window is mapped to a linalg.pooling operation. The type of
/// the pooling is determined based on the body of the reduce window
/// operation. This class enumerates the different variants.
enum class PoolingType {
  kMin,
  kMax,
  kAdd,
};

struct ReduceWindowOpConversion
    : public ConvertToLinalgBufferOp<ReduceWindowOpConversion,
                                     xla_hlo::ReduceWindowOp> {
  using ConvertToLinalgBufferOp<
      ReduceWindowOpConversion,
      xla_hlo::ReduceWindowOp>::ConvertToLinalgBufferOp;

  LogicalResult apply(xla_hlo::ReduceWindowOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;
};
}  // namespace

static PoolingType getPoolingType(Region &region) {
  assert(region.getBlocks().size() == 1 &&
         "expected the region has exactlly one block");
  Block &block = region.front();
  assert(block.getOperations().size() == 2 &&
         "expected the block has exactlly two operations");
  auto op = block.begin();
  if (isa<xla_hlo::MinOp>(op)) return PoolingType::kMin;
  if (isa<xla_hlo::MaxOp>(op)) return PoolingType::kMax;
  if (isa<xla_hlo::AddOp>(op)) return PoolingType::kAdd;

  llvm_unreachable("unknown pooling type");
}

LogicalResult ReduceWindowOpConversion::apply(
    xla_hlo::ReduceWindowOp op, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // Create a fake window dimension.
  SmallVector<int64_t, 4> shapes;
  for (auto dim : op.window_dimensions().getValues<int64_t>())
    shapes.push_back(dim);
  Type type = rewriter.getIntegerType(32);
  auto memrefType = MemRefType::get(shapes, type);
  auto fakeWindowDims = rewriter.create<AllocOp>(loc, memrefType);

  llvm::SmallVector<Attribute, 4> strides;
  if (op.window_strides().hasValue()) {
    strides.insert(strides.begin(),
                   op.window_strides().getValue().getAttributeValues().begin(),
                   op.window_strides().getValue().getAttributeValues().end());
  }
  auto stridesArg = ArrayAttr::get(strides, op.getContext());

  // TODO(hanchung): Use template lambda after migrating to C++20.
  auto createOp = [&](auto *type_ptr) -> linalg::LinalgOp {
    return cast<linalg::LinalgOp>(
        rewriter
            .create<std::remove_pointer_t<decltype(type_ptr)>>(
                loc, ArrayRef<Type>{}, inputBuffers[0],
                fakeWindowDims.getResult(), resultBuffers[0], stridesArg,
                /*dilations=*/nullptr,
                /*padding=*/nullptr)
            .getOperation());
  };
  linalg::LinalgOp poolingOp;
  PoolingType poolingType = getPoolingType(op.body());

  if (failed(zeroFillBuffer(loc, resultBuffers[0], rewriter))) {
    rewriter.notifyMatchFailure(op, "failed to zero fill result buffer");
    return failure();
  }
  switch (poolingType) {
    case PoolingType::kMin: {
      poolingOp = createOp(static_cast<linalg::PoolingMinOp *>(nullptr));
      break;
    }
    case PoolingType::kMax: {
      poolingOp = createOp(static_cast<linalg::PoolingMaxOp *>(nullptr));
      break;
    }
    case PoolingType::kAdd: {
      poolingOp = createOp(static_cast<linalg::PoolingSumOp *>(nullptr));
      break;
    }
  }

  rewriter.create<DeallocOp>(loc, fakeWindowDims);

  return success();
}

//===----------------------------------------------------------------------===//
// xla_hlo.reduce conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

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
    : public ConvertToLinalgBufferOp<ReduceOpConversion, xla_hlo::ReduceOp> {
  using ConvertToLinalgBufferOp<ReduceOpConversion,
                                xla_hlo::ReduceOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(xla_hlo::ReduceOp reduceOp, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;

 private:
  ReduceRegionTypeConverter converter;
};

/// Base class for converting operations within the reduction op region. Derived
/// classes implement the following static method to implement the conversion.
///
///   static Operation *apply(OpTy op, ArrayRef<Value> operands,
///                           ConversionPatternRewriter &rewriter);
template <typename DerivedTy, typename OpTy>
struct ReduceRegionOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only convert it if it is within a reduce op region.
    if (!isWithinReduceOpRegion(op)) return failure();
    Operation *replacement = DerivedTy::apply(op, operands, rewriter);
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
  static Operation *apply(OpTy op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter) {
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
  static Operation *apply(xla_hlo::ReturnOp op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter) {
    return rewriter.create<linalg::YieldOp>(op.getLoc(), operands[0]);
  }
};
}  // namespace

LogicalResult ReduceOpConversion::apply(
    xla_hlo::ReduceOp reduceOp, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  if (reduceOp.getNumOperands() != 2) return failure();
  Value src = *reduceOp.operands().begin();
  Value initVal = *reduceOp.init_values().begin();
  if (reduceOp.getNumResults() != 1) return failure();

  auto srcArgType = src.getType().template cast<ShapedType>();
  unsigned nInputRank = srcArgType.getRank();
  if (!nInputRank) return failure();

  // Get the reduction dimension. For now expects only a single reduction
  // dimension.
  auto loc = reduceOp.getLoc();
  DenseIntElementsAttr dimensionsAttr = reduceOp.dimensions();
  SmallVector<int, 4> reductionDims;
  for (const auto &dim : dimensionsAttr.getIntValues())
    reductionDims.push_back(dim.getSExtValue());

  // Check if initVal is constant. If so, inline the value into the region.
  Attribute initConstVal = getInitValueAsConst(initVal);
  if (initConstVal) {
    if (initVal.hasOneUse()) rewriter.eraseOp(initVal.getDefiningOp());
    initVal = rewriter.create<ConstantOp>(initVal.getDefiningOp()->getLoc(),
                                          initConstVal);
  }

  // Prepare indexing maps for linalg generic op. The elements are for src,
  // initial value and dst, respectively.
  // Transpose `src` to make the reduction loops be the innermost, because it's
  // easier to fully utilize processors.
  SmallVector<Attribute, 3> indexingMaps;
  indexingMaps.emplace_back(AffineMapAttr::get(getTransposeMapForReduction(
      rewriter.getContext(), nInputRank, reductionDims)));
  if (!initConstVal)
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
          : AffineMap::get(nInputRank, /*symbolCount=*/0, exprs,
                           rewriter.getContext())));

  SmallVector<Type, 2> resultTypes = {};
  SmallVector<Value, 2> linalgOpArgs = {inputBuffers[0]};
  if (!initConstVal) linalgOpArgs.push_back(inputBuffers[1]);
  linalgOpArgs.push_back(resultBuffers[0]);
  if (failed(zeroFillBuffer(loc, resultBuffers[0], rewriter))) {
    rewriter.notifyMatchFailure(reduceOp, "failed to zero fill result buffer");
    return failure();
  }
  auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
      loc, resultTypes, linalgOpArgs,
      rewriter.getI64IntegerAttr(linalgOpArgs.size() - 1),  // args_in
      rewriter.getI64IntegerAttr(1),                        // args_out
      rewriter.getArrayAttr(indexingMaps),
      getParallelAndReductionIterAttrs(rewriter, nInputRank,
                                       reductionDims.size()),
      /*doc=*/nullptr, /*library_call=*/nullptr);

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
    if (!initConstVal) signatureConverter.addInputs(convertedType);
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
    Value initArg =
        initConstVal ? initVal : entryBlock->getArgument(numArgs - 2);
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
  return success();
}

//===----------------------------------------------------------------------===//
// Linalg op on tensors to linalg op on buffers conversion base class.
//===----------------------------------------------------------------------===//

namespace {
template <typename LinalgOpTy>
struct LinalgOpOnTensorConversion
    : public ConvertToLinalgBufferOp<LinalgOpOnTensorConversion<LinalgOpTy>,
                                     LinalgOpTy> {
  using ConvertToLinalgBufferOp<LinalgOpOnTensorConversion<LinalgOpTy>,
                                LinalgOpTy>::ConvertToLinalgBufferOp;
  LogicalResult apply(LinalgOpTy op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const {
    if (!op.hasTensorSemantics()) return failure();
    SmallVector<Value, 2> opArgs(inputBuffers.begin(), inputBuffers.end());
    opArgs.append(resultBuffers.begin(), resultBuffers.end());

    // Create a new op with the same traits as the original
    // generic/indexed_generic op, but with memrefs.
    // TODO(ravishankarm): Figure out how to do this inplace.
    auto linalgBufferOp = rewriter.template create<LinalgOpTy>(
        op.getLoc(), ArrayRef<Type>(), opArgs, op.args_in(), op.args_out(),
        op.indexing_maps(), op.iterator_types(),
        /*doc=*/nullptr,
        /*library_call=*/nullptr);
    // Move the region from the replaced op into the new op.
    unsigned numTensorOperands = op.getNumOperands();
    // indexed_generic op has arguments for each index. In the case of generic
    // op, `numIndices` is zero.
    unsigned numIndices =
        op.region().begin()->getNumArguments() - numTensorOperands;
    auto &region = linalgBufferOp.region();
    region.takeBody(op.region());
    // Need to convert the signature to take extra arguments for the return
    // type.
    TypeConverter::SignatureConversion signatureConverter(numIndices +
                                                          numTensorOperands);
    for (int i = 0; i < numIndices; ++i)
      signatureConverter.addInputs(i, rewriter.getIndexType());
    for (auto arg : llvm::enumerate(opArgs)) {
      if (arg.index() < numTensorOperands) {
        signatureConverter.addInputs(
            numIndices + arg.index(),
            arg.value().getType().cast<MemRefType>().getElementType());
      } else {
        signatureConverter.addInputs(
            arg.value().getType().cast<MemRefType>().getElementType());
      }
    }
    rewriter.applySignatureConversion(&region, signatureConverter);
    return success();
  }
};

/// Convert linalg.tensor_reshape to linalg.reshape. The former has copy
/// semantics while the later is an aliasing instruction. As long as the operand
/// to the tensor_reshape has a single use, this distinction can be ignored.
struct TensorReshapeOpConversion
    : public OpConversionPattern<linalg::TensorReshapeOp> {
  TensorReshapeOpConversion(MLIRContext *context,
                            TensorToBufferMap const &resultTensorToBufferMap,
                            PatternBenefit benefit = 1)
      : OpConversionPattern<linalg::TensorReshapeOp>(context, benefit),
        resultTensorToBufferMap(resultTensorToBufferMap) {}

  LogicalResult matchAndRewrite(
      linalg::TensorReshapeOp reshapeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    linalg::TensorReshapeOp::Adaptor adaptor(operands);
    if (Value buffer =
            resolveResult(reshapeOp.src(), adaptor.src(), reshapeOp.result(),
                          resultTensorToBufferMap))
      rewriter.replaceOp(reshapeOp, buffer);
    else
      rewriter.replaceOpWithNewOp<linalg::ReshapeOp>(
          reshapeOp, getMemrefTypeForTensor(reshapeOp.result()), adaptor.src(),
          reshapeOp.reassociation());
    return success();
  }

 private:
  TensorToBufferMap const &resultTensorToBufferMap;
};
}  // namespace

//===----------------------------------------------------------------------===//
// hal.interface.*.tensor and shapex.* conversion.
//===----------------------------------------------------------------------===//

namespace {

/// Conversion for a shapex.tie_shape op on tensors to that on buffers. The
/// converted operation uses the same shape information.
struct ShapeOpPattern final : public OpConversionPattern<Shape::TieShapeOp> {
  ShapeOpPattern(MLIRContext *context,
                 TensorToBufferMap const &resultTensorToBufferMap,
                 PatternBenefit benefit = 1)
      : OpConversionPattern<Shape::TieShapeOp>(context, benefit),
        resultTensorToBufferMap(resultTensorToBufferMap) {}

  LogicalResult matchAndRewrite(
      Shape::TieShapeOp shapeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Shape::TieShapeOp::Adaptor adaptor(operands);
    if (Value buffer = resolveResult(shapeOp.operand(), adaptor.operand(),
                                     shapeOp.result(), resultTensorToBufferMap))
      rewriter.replaceOp(shapeOp, buffer);
    else
      rewriter.replaceOpWithNewOp<Shape::TieShapeOp>(
          shapeOp, getMemrefTypeForTensor(shapeOp.result()), adaptor.operand(),
          adaptor.shape());
    return success();
  }

 private:
  TensorToBufferMap const &resultTensorToBufferMap;
};

/// Replaces all uses hal.interface.load.tensor with iree.placeholder.
struct HALInterfaceLoadTensorOpEraser final
    : public OpConversionPattern<IREE::HAL::InterfaceLoadTensorOp> {
  HALInterfaceLoadTensorOpEraser(
      MLIRContext *context, TensorToBufferMap const &resultTensorToBufferMap,
      PatternBenefit benefit = 1)
      : OpConversionPattern<IREE::HAL::InterfaceLoadTensorOp>(context, benefit),
        resultTensorToBufferMap(resultTensorToBufferMap) {}

  LogicalResult matchAndRewrite(IREE::HAL::InterfaceLoadTensorOp loadOp,
                                ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
    if (!matchPattern(loadOp.offset(), m_Zero()))
      return loadOp.emitError("unhandled non-zero offset");

    // Get the corresponding memref type from the tensor type.
    auto tensorType = loadOp.result().getType().cast<RankedTensorType>();
    auto bindingOp = loadOp.queryBindingOp();
    assert(bindingOp);
    auto bufferType = getTensorBackingBufferType(tensorType, bindingOp.type());

    // Create the placeholder op for the backing buffer. Make sure shape
    // annotation is carried over if exists.
    auto phOp = rewriter.create<IREE::PlaceholderOp>(
        loadOp.getLoc(), bufferType, "interface buffer");
    phOp.setAttr("binding", loadOp.binding());
    Value buffer = phOp.getResult();

    // If the result of the load is already mapped to a buffer, a copy is
    // required from the buffer above into the mapped buffer. This happens when
    // in the original computation the loaded tensor value goes through a chain
    // of view-like operations and is used as an operand to a store tensor
    // operation.
    if (Value outputBuffer = resultTensorToBufferMap.lookup(loadOp.result())) {
      rewriter.create<linalg::CopyOp>(loadOp.getLoc(), buffer, outputBuffer);
      rewriter.replaceOp(loadOp, outputBuffer);
    } else {
      rewriter.replaceOp(loadOp, buffer);
    }
    return success();
  }

 private:
  TensorToBufferMap const &resultTensorToBufferMap;
};

/// Erases the hal.interface.store.tensor and replace all uses with the buffer.
struct HALInterfaceStoreTensorOpEraser final
    : public OpConversionPattern<IREE::HAL::InterfaceStoreTensorOp> {
  HALInterfaceStoreTensorOpEraser(
      MLIRContext *context, TensorToBufferMap const &resultTensorToBufferMap,
      PatternBenefit benefit = 1)
      : OpConversionPattern<IREE::HAL::InterfaceStoreTensorOp>(context,
                                                               benefit),
        resultTensorToBufferMap(resultTensorToBufferMap) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceStoreTensorOp storeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::InterfaceStoreTensorOp::Adaptor adaptor(operands);
    Value operand = adaptor.operand();
    // If we are just storing the buffer back to itself again, we can trivially
    // remove this op. Otherwise, copy the content from the source buffer to the
    // destination buffer.
    Value outputBuffer = resultTensorToBufferMap.lookup(storeOp.operand());
    if (outputBuffer == operand) {
      rewriter.eraseOp(storeOp);
      return success();
    }
    if (outputBuffer) {
      rewriter.replaceOpWithNewOp<linalg::CopyOp>(storeOp, operand,
                                                  outputBuffer);
      return success();
    }
    return failure();
  }

 private:
  TensorToBufferMap const &resultTensorToBufferMap;
};
}  // namespace

/// When converting all tensor-based ops to buffer-based ops, Instead of
/// creating a tensor value that is stored into memory using
/// hal.interface.store.tensor, a buffer is needed into which the operations
/// that computes the result will write into directly. Create these buffers
/// using a iree.placeholder instruction that return the memref view of a
/// interface buffer. These are added at the start of the function so that any
/// operation that needs to write into this buffer can use it and maintain SSA
/// property of the buffer. The map `resultTensorToBufferMap` is updated to
/// associate the tensor value that is stored with the buffer created. So when
/// that value is seen during lowering the correct result buffer is used.
///
/// There might be a sequence of view-like operations on memref, which dont
/// modify the buffer, but just the way they are referenced. For example,
///
/// %a = linalg.tensor_reshape %tensor [..] : tensor<typeA> into tensor<typeB>
/// %b = shapex.tie_shape %a, ... : tensor<typeB> ...
/// hal.interface.store.tensor %b ... : tensor<typeB>
///
/// When converted to buffers these instructions need to be replayed "in
/// reverse" to get the buffer to use as replacement.
///
/// %b = iree.placeholder ... : memref<typeB>
/// %a = shapex.tie_shape %b, ... : memref<typeB>
/// %buffer = linalg.reshape %a [..] : memref<typeB> into memref<typeA>
///
/// For each of the view-like operations, mark the tensor to buffer conversion
/// as resolved and associate the source of the view operand with the
/// corresponding result buffer.
///
/// Note : The tensor_reshape op is also treated as a view-like operation, while
/// in reality its semantics is a copy semantics. As long as the operand for the
/// tensor_reshape operation has a single use (the tensor_reshape) there
/// distinction can be ignored.
static LogicalResult createAndPropagateBufferUsedForResultTensor(
    IREE::HAL::InterfaceStoreTensorOp op,
    TensorToBufferMap &resultTensorToBufferMap, OpBuilder &builder) {
  if (!matchPattern(op.offset(), m_Zero()))
    return op.emitError("unhandled non-zero offset");

  // Get the corresponding memref type from the tensor type.
  Value tensor = op.operand();
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  auto bindingOp = op.queryBindingOp();
  assert(bindingOp);
  auto bufferType = getTensorBackingBufferType(tensorType, bindingOp.type());

  // Create the placeholder op for the backing buffer. Make sure shape
  // annotation is carried over if exists.
  auto phOp = builder.create<IREE::PlaceholderOp>(op.getLoc(), bufferType,
                                                  "interface buffer");
  phOp.setAttr("binding", op.binding());
  Value buffer = phOp;
  resultTensorToBufferMap[tensor] = buffer;

  while (true) {
    if (auto tieShapeOp = tensor.getDefiningOp<Shape::TieShapeOp>()) {
      if (!tieShapeOp.result().hasOneUse()) break;
      builder.setInsertionPointAfter(tieShapeOp.shape().getDefiningOp());
      auto newTieShapeOp = builder.create<Shape::TieShapeOp>(
          op.getLoc(), buffer.getType(), buffer, tieShapeOp.shape());
      tensor = tieShapeOp.operand();
      buffer = newTieShapeOp.result();
      resultTensorToBufferMap[tensor] = buffer;
      continue;
    }
    if (auto tensorReshapeOp =
            tensor.getDefiningOp<linalg::TensorReshapeOp>()) {
      if (!tensorReshapeOp.result().hasOneUse()) break;
      auto newReshapeOp = builder.create<linalg::ReshapeOp>(
          op.getLoc(), getMemrefTypeForTensor(tensorReshapeOp.getSrcType()),
          buffer, tensorReshapeOp.reassociation());
      tensor = tensorReshapeOp.src();
      buffer = newReshapeOp.result();
      resultTensorToBufferMap[tensor] = buffer;
      continue;
    }
    break;
  }
  return success();
}

/// Processes the hal.interface.store.tensor instructions to get buffer views
/// for the inputs/outputs to the dispatch function.
static LogicalResult createAndPropagateBufferUsedForResultTensors(
    FuncOp funcOp, TensorToBufferMap &resultTensorToBufferMap) {
  OpBuilder builder(funcOp.getBody());
  auto walkResult = funcOp.walk(
      [&](IREE::HAL::InterfaceStoreTensorOp storeTensorOp) -> WalkResult {
        return createAndPropagateBufferUsedForResultTensor(
            storeTensorOp, resultTensorToBufferMap, builder);
      });
  return failure(walkResult.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Pass specification.
//===----------------------------------------------------------------------===//

namespace {
struct ConvertHLOToLinalgOnBuffersPass
    : public PassWrapper<ConvertHLOToLinalgOnBuffersPass, FunctionPass> {
  void runOnFunction() override;
};
}  // namespace

void populateHLOToLinalgOnBuffersConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    TensorToBufferMap const &resultTensorToBufferMap) {
  patterns
      .insert<ConvOpConversion,
              DotOpConversion<DotOperationType::MatrixMatrix, linalg::MatmulOp>,
              LinalgOpOnTensorConversion<linalg::GenericOp>,
              LinalgOpOnTensorConversion<linalg::IndexedGenericOp>,
              PadOpConversion, ReduceOpConversion, ReduceWindowOpConversion,
              TensorReshapeOpConversion, TorchIndexSelectOpConversion>(
          context, resultTensorToBufferMap);
  // Reduce region operation conversions.
  patterns.insert<ReduceRegionXLAOpConversion<xla_hlo::AddOp>,
                  ReduceRegionXLAOpConversion<xla_hlo::MinOp>,
                  ReduceRegionXLAOpConversion<xla_hlo::MaxOp>,
                  ReduceRegionReturnOpConversion>(context);
}

void ConvertHLOToLinalgOnBuffersPass::runOnFunction() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getFunction();

  // First create buffers for all StoreTensorOps.
  TensorToBufferMap resultTensorToBufferMap;
  if (failed(createAndPropagateBufferUsedForResultTensors(
          funcOp, resultTensorToBufferMap)))
    return signalPassFailure();

  OwningRewritePatternList patterns;
  populateHLOToLinalgOnBuffersConversionPatterns(context, patterns,
                                                 resultTensorToBufferMap);
  patterns.insert<HALInterfaceLoadTensorOpEraser,
                  HALInterfaceStoreTensorOpEraser, ShapeOpPattern>(
      context, resultTensorToBufferMap);

  ConversionTarget target(*context);
  // Make sure all XLA HLO ops are converted to Linalg ops after this pass.
  target.addIllegalDialect<xla_hlo::XlaHloDialect>();
  // All Linalg ops should operate on buffers. So hal.interface.*.tensor ops
  // should be gone.
  target.addIllegalOp<IREE::HAL::InterfaceLoadTensorOp,
                      IREE::HAL::InterfaceStoreTensorOp>();
  target.addDynamicallyLegalOp<Shape::TieShapeOp>(
      [](Shape::TieShapeOp op) -> bool {
        return op.operand().getType().isa<MemRefType>();
      });
  // Also convert away linalg.tensor_reshape.
  target.addIllegalOp<linalg::TensorReshapeOp>();
  target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
      Optional<ConversionTarget::DynamicLegalityCallbackFn>([](Operation *op) {
        // The generated structured Linalg ops should have buffer semantics.
        if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
          return linalgOp.hasBufferSemantics();
        // The other Linalg ops (like linalg.yield) are okay.
        return true;
      }));
  // Let the rest fall through.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyFullConversion(getFunction(), target, patterns))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnBuffersPass() {
  return std::make_unique<ConvertHLOToLinalgOnBuffersPass>();
}

static PassRegistration<ConvertHLOToLinalgOnBuffersPass> pass(
    "iree-codegen-hlo-to-linalg-on-buffers",
    "Convert from XLA-HLO ops to Linalg ops on buffers");
}  // namespace iree_compiler
}  // namespace mlir
