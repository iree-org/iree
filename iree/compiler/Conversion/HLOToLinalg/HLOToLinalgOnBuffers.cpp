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
// where the dispatch region contains a single mhlo op that can be converted
// to linalg on buffers.
//
//===----------------------------------------------------------------------===//

#include <cstddef>

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

using OutputBufferMap = DenseMap<Operation *, Value>;

// -----------------------------------------------------------------------------
// Utility functions.
// -----------------------------------------------------------------------------

/// Returns the constant value associated with the init value if the defining
/// operation is a constant.
static Attribute getInitValueAsConst(Value init) {
  DenseElementsAttr attr;
  if (!matchPattern(init, m_Constant(&attr))) return {};
  auto type = attr.getType().dyn_cast<ShapedType>();
  if (!type || type.getRank() != 0) return {};
  if (auto intType = type.getElementType().dyn_cast<IntegerType>()) {
    return IntegerAttr::get(intType, attr.getValue<APInt>({}));
  } else if (auto floatType = type.getElementType().dyn_cast<FloatType>()) {
    return FloatAttr::get(floatType, attr.getValue<APFloat>({}));
  }
  return {};
}

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
static SmallVector<StringRef, 3> getParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction) {
  SmallVector<StringRef, 3> res(nLoops - nReduction,
                                getParallelIteratorTypeName());
  res.append(nReduction, getReductionIteratorTypeName());
  return res;
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
    default:
      llvm_unreachable("unexpected descriptor type");
  }
}

/// Returns the MemRefType to use for a given `tensorType`.
static MemRefType getMemrefTypeForTensor(
    ShapedType tensorType, ArrayRef<AffineMap> affineMapComposition = {},
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
        if (auto shapedType = result.value().getType().dyn_cast<ShapedType>()) {
          if (shapedType.hasStaticShape()) {
            resultBuffer = rewriter.create<AllocOp>(
                op->getLoc(), getMemrefTypeForTensor(shapedType));
          }
        }
      }
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
// mhlo.dot conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
enum class DotOperationType {
  VectorDot = 0,
  MatrixVector = 1,
  MatrixMatrix = 2,
  Unsupported = 3
};
}

static DotOperationType getDotOperationType(mhlo::DotOp dotOp) {
  ArrayRef<int64_t> lhsShape =
      dotOp.lhs().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape =
      dotOp.rhs().getType().cast<ShapedType>().getShape();
  auto shapeMatches = [](int64_t a, int64_t b) {
    return a == ShapedType::kDynamicSize || b == ShapedType::kDynamicSize ||
           a == b;
  };
  if (lhsShape.size() == 1 && rhsShape.size() == 1 &&
      shapeMatches(lhsShape[0], rhsShape[0])) {
    return DotOperationType::VectorDot;
  }
  if (lhsShape.size() == 2 && rhsShape.size() == 1 &&
      shapeMatches(lhsShape[1], rhsShape[0])) {
    return DotOperationType::MatrixVector;
  }
  if (rhsShape.size() == 2 && rhsShape.size() == 2 &&
      shapeMatches(lhsShape[1], rhsShape[0])) {
    return DotOperationType::MatrixMatrix;
  }
  return DotOperationType::Unsupported;
}

namespace {
/// Converts mhlo.dot operation to linalg.matmul op
template <DotOperationType opType, typename LinalgOpTy>
struct DotOpConversion
    : public ConvertToLinalgBufferOp<DotOpConversion<opType, LinalgOpTy>,
                                     mhlo::DotOp> {
  using ConvertToLinalgBufferOp<DotOpConversion<opType, LinalgOpTy>,
                                mhlo::DotOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(mhlo::DotOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const {
    if (getDotOperationType(op) == opType) {
      if (failed(zeroFillBuffer(op.getLoc(), resultBuffers[0], rewriter))) {
        rewriter.notifyMatchFailure(op, "failed to zero fill result buffer");
        return failure();
      }
      rewriter.create<LinalgOpTy>(op.getLoc(), inputBuffers, resultBuffers);
      return success();
    }
    return failure();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.dot_general conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.dot_general operation to linalg.batchmatmul op
struct DotGeneralOpConversion
    : public ConvertToLinalgBufferOp<DotGeneralOpConversion,
                                     mhlo::DotGeneralOp> {
  using ConvertToLinalgBufferOp<DotGeneralOpConversion,
                                mhlo::DotGeneralOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(mhlo::DotGeneralOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const {
    auto extract1DVector = [](DenseIntElementsAttr elements) {
      SmallVector<int64_t, 6> ret;
      for (const APInt &element : elements) {
        ret.push_back(element.getLimitedValue());
      }
      return ret;
    };
    mhlo::DotDimensionNumbers dimNumbers = op.dot_dimension_numbers();
    auto lhsBatchingDims =
        extract1DVector(dimNumbers.lhs_batching_dimensions());
    auto rhsBatchingDims =
        extract1DVector(dimNumbers.rhs_batching_dimensions());
    auto lhsContractingDims =
        extract1DVector(dimNumbers.lhs_contracting_dimensions());
    auto rhsContractingDims =
        extract1DVector(dimNumbers.rhs_contracting_dimensions());
    if (lhsBatchingDims.size() != 1 || lhsBatchingDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs batching dimensions exactly {0}");
    }
    if (rhsBatchingDims.size() != 1 || rhsBatchingDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs batching dimensions exactly {0}");
    }
    if (lhsContractingDims.size() != 1 || lhsContractingDims[0] != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs contracting dimensions exactly {2}");
    }
    if (rhsContractingDims.size() != 1 || rhsContractingDims[0] != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs contracting dimensions exactly {1}");
    }
    if (failed(zeroFillBuffer(op.getLoc(), resultBuffers[0], rewriter))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to zero fill result buffer");
    }
    rewriter.create<linalg::BatchMatmulOp>(op.getLoc(), inputBuffers,
                                           resultBuffers);
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.convolution conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.convolution operation to linalg.conv op.
struct ConvOpConversion
    : public ConvertToLinalgBufferOp<ConvOpConversion, mhlo::ConvOp> {
  using ConvertToLinalgBufferOp<ConvOpConversion,
                                mhlo::ConvOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(mhlo::ConvOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;
};
}  // namespace

LogicalResult ConvOpConversion::apply(
    mhlo::ConvOp op, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  if (const auto dimensionNumbers = op.dimension_numbers()) {
    const int inputSpatialRank =
        llvm::size(dimensionNumbers.input_spatial_dimensions());
    // The dimensions for input should follow the order of
    // batch_count, spatial_dims..., input_feature_count.
    if (dimensionNumbers.input_batch_dimension().getInt() != 0 ||
        dimensionNumbers.input_feature_dimension().getInt() !=
            (inputSpatialRank + 1)) {
      return failure();
    }

    const int kernelSpatialRank =
        llvm::size(dimensionNumbers.kernel_spatial_dimensions());
    // The dimensions for filter should follow the order of
    // spatial_dims..., input_feature_count, num_output_feature_count.
    if (dimensionNumbers.kernel_input_feature_dimension().getInt() !=
            kernelSpatialRank ||
        dimensionNumbers.kernel_output_feature_dimension().getInt() !=
            (kernelSpatialRank + 1)) {
      return failure();
    }

    const int outputSpatialRank =
        llvm::size(dimensionNumbers.output_spatial_dimensions());
    // The dimensions for output should follow the order of
    // batch_count, spatial_dims.., output_feature_count.
    if (dimensionNumbers.output_batch_dimension().getInt() != 0 ||
        dimensionNumbers.output_feature_dimension().getInt() !=
            (outputSpatialRank + 1)) {
      return failure();
    }

    if (inputSpatialRank != outputSpatialRank ||
        inputSpatialRank != kernelSpatialRank) {
      return failure();
    }

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
          (*kernelSpatialDim++).getZExtValue() != i) {
        return failure();
      }
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

  ShapedType filterShapeType =
      op.rhs().getType().dyn_cast_or_null<ShapedType>();
  if (!filterShapeType) return failure();
  auto shape = filterShapeType.getShape();
  auto numGroups =
      shape[op.dimension_numbers().kernel_input_feature_dimension().getInt()];
  auto groupSize =
      shape[op.dimension_numbers().kernel_output_feature_dimension().getInt()];
  // Depthwise conv path...
  if (op.feature_group_count() > 1u && op.feature_group_count() == numGroups) {
    // Lowering depthwise convolution to linalg.generic op. The idea is to use
    // the group convolution formulation to perform the separable depthwise
    // convolution as the following, given an n-dimensional input x and filter w
    // the direct convolution operation can be written as:
    //  y[n, d1, d2, ....dn, ci * groupSize + co] = sum(k1, k2, ....kn,
    // x[n, d1 * stride1 + k1, d1 * stride2 + k2, ...dn * striden + kn]
    // * w[k1, k2, ...kn, ci, co])

    // TODO(ataei): Support dilation.
    if (llvm::any_of(dilation, [](Attribute attr) {
          return (attr.dyn_cast<IntegerAttr>().getInt() != 1);
        })) {
      return failure();
    }

    SmallVector<AffineExpr, 4> inputExprs;
    SmallVector<AffineExpr, 4> filterExprs;
    SmallVector<AffineExpr, 4> outputExprs;

    const auto spatialDims =
        llvm::size(op.dimension_numbers().input_spatial_dimensions());
    const int d1Index = 1;
    const int coIndex = d1Index + spatialDims;
    const int ciIndex = coIndex + 1;
    const int k1Index = ciIndex + 1;
    // n, d1 * stride1 + k1, d1 * stride2 + k2, ...dn * striden + kn
    inputExprs.push_back(rewriter.getAffineDimExpr(0));
    for (int i = 0; i < spatialDims; ++i) {
      if (op.window_stridesAttr()) {
        auto stride = op.window_stridesAttr().getValue<APInt>(i);
        inputExprs.push_back(rewriter.getAffineDimExpr(d1Index + i) *
                                 stride.getZExtValue() +
                             rewriter.getAffineDimExpr(k1Index + i));
      } else {
        inputExprs.push_back(rewriter.getAffineDimExpr(d1Index + i) +
                             rewriter.getAffineDimExpr(k1Index + i));
      }
    }
    inputExprs.push_back(rewriter.getAffineDimExpr(ciIndex));

    // k1, k2, ...kn, ci, co
    for (int i = 0; i < spatialDims; ++i) {
      filterExprs.push_back(rewriter.getAffineDimExpr(k1Index + i));
    }
    filterExprs.push_back(rewriter.getAffineDimExpr(ciIndex));
    filterExprs.push_back(rewriter.getAffineDimExpr(coIndex));

    // n, d1, d2, ....dn, ci * groupSize + co
    outputExprs.push_back(rewriter.getAffineDimExpr(0));
    for (int i = 0; i < spatialDims; ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(d1Index + i));
    }
    outputExprs.push_back(rewriter.getAffineDimExpr(ciIndex) * groupSize +
                          rewriter.getAffineDimExpr(coIndex));

    // nloops = |d| + |k| + |{n, ci, co}|
    int nloops = spatialDims * 2 + 3;
    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.emplace_back(AffineMap::get(
        nloops, /*symbolCount=*/0, inputExprs, rewriter.getContext()));
    indexingMaps.emplace_back(AffineMap::get(
        nloops, /*symbolCount=*/0, filterExprs, rewriter.getContext()));
    indexingMaps.emplace_back(AffineMap::get(
        nloops, /*symbolCount=*/0, outputExprs, rewriter.getContext()));

    Location loc = op.getLoc();

    SmallVector<StringRef, 3> loopAttributeTypes(spatialDims + 3, "parallel");
    loopAttributeTypes.append(spatialDims, "reduction");
    rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/ArrayRef<Type>{},
        /*inputs=*/inputBuffers,
        /*outputs=*/resultBuffers, indexingMaps, loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value mul = nestedBuilder.create<MulFOp>(nestedLoc, args[0], args[1]);
          Value add = nestedBuilder.create<AddFOp>(nestedLoc, mul, args[2]);
          nestedBuilder.create<linalg::YieldOp>(loc, add);
        });
  } else {
    rewriter.create<linalg::ConvOp>(op.getLoc(), inputBuffers[1],
                                    inputBuffers[0], resultBuffers[0],
                                    stridesArg, dilationArg, padding);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// mhlo.concatenate conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Converts a mhlo.concatenate op to subview ops + linalg.copy/fill ops.
class ConcatenateOpConversion
    : public ConvertToLinalgBufferOp<ConcatenateOpConversion,
                                     mhlo::ConcatenateOp> {
 public:
  using ConvertToLinalgBufferOp<ConcatenateOpConversion,
                                mhlo::ConcatenateOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(mhlo::ConcatenateOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;
};
}  // namespace

LogicalResult ConcatenateOpConversion::apply(
    mhlo::ConcatenateOp op, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  int dim = op.dimension();
  int rank = inputBuffers[0].getType().cast<ShapedType>().getRank();
  SmallVector<Value, 3> offsets, sizes, strides;
  for (int i = 0; i < rank; ++i) {
    offsets.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    Value size = rewriter.create<DimOp>(loc, resultBuffers[0], i);
    sizes.push_back(size);
    strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
  }

  Value accBound = rewriter.create<ConstantIndexOp>(loc, 0);
  for (auto inBuf : inputBuffers) {
    offsets[dim] = accBound;
    if (auto cstOp = inBuf.getDefiningOp<ConstantOp>()) {
      sizes[dim] = rewriter.create<ConstantIndexOp>(
          loc, cstOp.getType().cast<ShapedType>().getShape()[dim]);
      auto subViewOp = rewriter.create<SubViewOp>(loc, resultBuffers[0],
                                                  offsets, sizes, strides);
      auto inputConstAttr =
          cstOp.valueAttr().cast<DenseElementsAttr>().getSplatValue();
      Value cstVal = rewriter.create<ConstantOp>(loc, inputConstAttr);
      rewriter.create<linalg::FillOp>(loc, subViewOp, cstVal);
    } else {
      sizes[dim] = rewriter.create<DimOp>(loc, inBuf, dim);
      auto subViewOp = rewriter.create<SubViewOp>(loc, resultBuffers[0],
                                                  offsets, sizes, strides);
      rewriter.create<linalg::CopyOp>(loc, inBuf, subViewOp);
    }
    accBound = rewriter.create<AddIOp>(loc, accBound, sizes[dim]);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// mhlo.pad conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.pad operation to linalg.indexed_generic op.
// TODO(#1604): Lower the pad op to a Linalg named op.
struct PadOpConversion
    : public ConvertToLinalgBufferOp<PadOpConversion, mhlo::PadOp> {
  using ConvertToLinalgBufferOp<PadOpConversion,
                                mhlo::PadOp>::ConvertToLinalgBufferOp;

  LogicalResult apply(mhlo::PadOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;
};
}  // namespace

LogicalResult PadOpConversion::apply(
    mhlo::PadOp op, ArrayRef<Value> inputBuffers, ArrayRef<Value> resultBuffers,
    ConversionPatternRewriter &rewriter) const {
  mhlo::PadOp::Adaptor adaptor(inputBuffers);
  auto loc = op.getLoc();

  Attribute paddingConstVal = getInitValueAsConst(adaptor.padding_value());
  Value paddingVal =
      paddingConstVal
          ? rewriter.create<ConstantOp>(loc, paddingConstVal).getResult()
          : rewriter.create<LoadOp>(loc, adaptor.padding_value());

  const auto &edgePaddingLow = op.edge_padding_low();
  const auto &interiorPadding = op.interior_padding();
  SmallVector<Value, 3> offsets, sizes, strides;
  for (auto it : llvm::enumerate(llvm::zip(edgePaddingLow, interiorPadding))) {
    Value startIndex = rewriter.create<ConstantIndexOp>(
        loc, std::get<0>(it.value()).getZExtValue());
    offsets.push_back(startIndex);
    Value size = rewriter.create<DimOp>(loc, inputBuffers[0], it.index());
    sizes.push_back(size);
    Value stride = rewriter.create<ConstantIndexOp>(
        loc, std::get<1>(it.value()).getZExtValue() + 1);
    strides.push_back(stride);
  }

  rewriter.create<linalg::FillOp>(loc, resultBuffers[0], paddingVal);
  auto subViewOp = rewriter.create<SubViewOp>(loc, resultBuffers[0], offsets,
                                              sizes, strides);
  if (auto cstOp = dyn_cast<ConstantOp>(inputBuffers[0].getDefiningOp())) {
    auto inputConstAttr =
        cstOp.valueAttr().cast<DenseElementsAttr>().getSplatValue();
    Value cstVal = rewriter.create<ConstantOp>(loc, inputConstAttr);
    rewriter.create<linalg::FillOp>(loc, subViewOp, cstVal);
  } else {
    rewriter.create<linalg::CopyOp>(loc, inputBuffers[0], subViewOp);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// mhlo.slice conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.slice operation to linalg.subview + linalg.copy
struct SliceOpConversion : public OpConversionPattern<mhlo::SliceOp> {
  SliceOpConversion(MLIRContext *context,
                    TensorToBufferMap const &resultTensorToBufferMap,
                    PatternBenefit benefit = 1)
      : OpConversionPattern<mhlo::SliceOp>(context, benefit),
        resultTensorToBufferMap(resultTensorToBufferMap) {}

  LogicalResult matchAndRewrite(
      mhlo::SliceOp op, ArrayRef<Value> inputBuffers,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto argType = inputBuffers[0].getType().template dyn_cast<ShapedType>();
    if (!argType || !argType.hasStaticShape()) {
      return op.emitError("expected static shape");
    }

    auto resultShape = op.getResult().getType().cast<ShapedType>().getShape();
    SmallVector<Value, 3> offsets, sizes, strides;
    for (int i = 0, e = argType.getRank(); i < e; ++i) {
      Value startIndex = rewriter.create<ConstantIndexOp>(
          loc, op.start_indices().getValue<int64_t>(i));
      offsets.push_back(startIndex);
      Value size = rewriter.create<ConstantIndexOp>(loc, resultShape[i]);
      sizes.push_back(size);
      Value stride = rewriter.create<ConstantIndexOp>(
          loc, op.strides().getValue<int64_t>(i));
      strides.push_back(stride);
    }
    auto subViewOp = rewriter.create<SubViewOp>(loc, inputBuffers[0], offsets,
                                                sizes, strides);

    // If the result of the subview is already mapped to a buffer, a copy is
    // required from the buffer above into the mapped buffer.
    if (Value bufferForResult =
            resultTensorToBufferMap.lookup(op.getResult())) {
      rewriter.create<linalg::CopyOp>(loc, subViewOp, bufferForResult);
      rewriter.replaceOp(op, bufferForResult);
    } else {
      rewriter.replaceOp(op, subViewOp.getResult());
    }

    return success();
  }

 private:
  TensorToBufferMap const &resultTensorToBufferMap;
};
}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.reduce_window conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {

/// mhlo.reduce_window is mapped to a linalg.pooling operation. The type of
/// the pooling is determined based on the body of the reduce window
/// operation. This class enumerates the different variants.
enum class PoolingType {
  kMin,
  kMax,
  kAdd,
};

struct ReduceWindowOpConversion
    : public ConvertToLinalgBufferOp<ReduceWindowOpConversion,
                                     mhlo::ReduceWindowOp> {
  using ConvertToLinalgBufferOp<ReduceWindowOpConversion,
                                mhlo::ReduceWindowOp>::ConvertToLinalgBufferOp;

  LogicalResult apply(mhlo::ReduceWindowOp op, ArrayRef<Value> inputBuffers,
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
  if (isa<mhlo::MinOp>(op)) return PoolingType::kMin;
  if (isa<mhlo::MaxOp>(op)) return PoolingType::kMax;
  if (isa<mhlo::AddOp>(op)) return PoolingType::kAdd;

  llvm_unreachable("unknown pooling type");
}

LogicalResult ReduceWindowOpConversion::apply(
    mhlo::ReduceWindowOp op, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // Create a fake window dimension.
  SmallVector<int64_t, 4> shapes;
  for (auto dim : op.window_dimensions().getValues<int64_t>()) {
    shapes.push_back(dim);
  }
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
// mhlo.reduce conversion patterns and utility functions.
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
  for (int i = 0; i < rank; ++i) {
    if (!s.count(i)) permutation.push_back(i);
  }
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

/// Type converter for converting the region of an mhlo::reduce op.
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

/// Converts the mhlo.reduce op on tensors to a linalg.indexed_generic op on
/// buffers. Expects that the reduce op is the only op within the dispatch
/// function. This pattern also fuses std.constant operations which are defining
/// ops of the init value with the linalg.indexed_generic op.
struct ReduceOpConversion
    : public ConvertToLinalgBufferOp<ReduceOpConversion, mhlo::ReduceOp> {
  using ConvertToLinalgBufferOp<ReduceOpConversion,
                                mhlo::ReduceOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(mhlo::ReduceOp reduceOp, ArrayRef<Value> inputBuffers,
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
    Value result = lmhlo::HloOpToStdScalarOp::map<OpTy>(
        op, operands[0].getType(), operands, &rewriter);
    return result.getDefiningOp();
  }
};

/// Converts mhlo.return to within a reduce region to a linalg.yield.
struct ReduceRegionReturnOpConversion final
    : public ReduceRegionOpConversion<ReduceRegionReturnOpConversion,
                                      mhlo::ReturnOp> {
  using ReduceRegionOpConversion<ReduceRegionReturnOpConversion,
                                 mhlo::ReturnOp>::ReduceRegionOpConversion;
  static Operation *apply(mhlo::ReturnOp op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter) {
    return rewriter.create<linalg::YieldOp>(op.getLoc(), operands[0]);
  }
};
}  // namespace

LogicalResult ReduceOpConversion::apply(
    mhlo::ReduceOp reduceOp, ArrayRef<Value> inputBuffers,
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
  for (const auto &dim : dimensionsAttr.getIntValues()) {
    reductionDims.push_back(dim.getSExtValue());
  }

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
  SmallVector<AffineMap, 3> indexingMaps;
  indexingMaps.emplace_back(getTransposeMapForReduction(
      rewriter.getContext(), nInputRank, reductionDims));
  if (!initConstVal) {
    indexingMaps.emplace_back(
        AffineMap::get(nInputRank, /*symbolCount=*/0, rewriter.getContext()));
  }
  // The indexing map of `dst` should drop the reduction loops. Since the
  // reduction loops now are all in the innermost, drops `reductionDims.size()`
  // dimensions. We don't need an inverse permutation here because they are the
  // same.
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0, e = nInputRank - reductionDims.size(); i < e; ++i) {
    exprs.push_back(rewriter.getAffineDimExpr(i));
  }
  indexingMaps.emplace_back(
      exprs.empty()
          ? AffineMap::get(nInputRank, /*symbolCount=*/0, rewriter.getContext())
          : AffineMap::get(nInputRank, /*symbolCount=*/0, exprs,
                           rewriter.getContext()));

  SmallVector<Type, 2> resultTypes = {};
  SmallVector<Value, 2> inputs = {inputBuffers[0]};
  if (!initConstVal) {
    inputs.push_back(inputBuffers[1]);
  }
  if (failed(zeroFillBuffer(loc, resultBuffers[0], rewriter))) {
    rewriter.notifyMatchFailure(reduceOp, "failed to zero fill result buffer");
    return failure();
  }
  auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
      loc, /*resultTensorTypes=*/resultTypes, /*inputs=*/inputs,
      /*outputBuffers=*/resultBuffers, indexingMaps,
      getParallelAndReductionIterators(nInputRank, reductionDims.size()));

  rewriter.inlineRegionBefore(reduceOp.body(), linalgOp.region(),
                              linalgOp.region().end());
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
    inputBuffers = inputBuffers.drop_back(op.getNumResults());
    SmallVector<Value, 2> opArgs = llvm::to_vector<2>(inputBuffers);
    opArgs.append(resultBuffers.begin(), resultBuffers.end());

    // Create a new op with the same traits as the original
    // generic/indexed_generic op, but with memrefs.
    // TODO(ravishankarm): Figure out how to do this inplace.
    auto linalgBufferOp = rewriter.template create<LinalgOpTy>(
        op.getLoc(), inputBuffers, resultBuffers,
        llvm::to_vector<4>(
            op.indexing_maps().template getAsValueRange<AffineMapAttr>()),
        llvm::to_vector<4>(
            op.iterator_types().template getAsValueRange<StringAttr>()));

    // Move the region from the replaced op into the new op.
    unsigned numTensorOperands = op.getNumOperands();
    // indexed_generic op has arguments for each index. In the case of generic
    // op, `numIndices` is zero.
    unsigned numIndices =
        op.region().begin()->getNumArguments() - numTensorOperands;
    auto &region = linalgBufferOp.region();
    rewriter.inlineRegionBefore(op.region(), region, region.end());
    // Need to convert the signature to take extra arguments for the return
    // type.
    TypeConverter::SignatureConversion signatureConverter(numIndices +
                                                          numTensorOperands);
    for (int i = 0; i < numIndices; ++i) {
      signatureConverter.addInputs(i, rewriter.getIndexType());
    }
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
    // If result has an associated buffer.
    Value bufferForResult = resultTensorToBufferMap.lookup(reshapeOp.result());
    if (!bufferForResult) {
      // This is not a reshape before store_tensor. Replace this op with a
      // reshape on buffers.
      rewriter.replaceOpWithNewOp<linalg::ReshapeOp>(
          reshapeOp, getMemrefTypeForTensor(reshapeOp.result()), adaptor.src(),
          reshapeOp.reassociation());
      return success();
    }

    // Look at all uses of bufferForResult in reshape ops. If once of those is
    // the input operand, there is nothing to do.
    if (!llvm::any_of(bufferForResult.getUses(), [&](auto &use) {
          auto bufferReshapeOp = dyn_cast<linalg::ReshapeOp>(use.getOwner());
          return bufferReshapeOp && bufferReshapeOp.result() == adaptor.src();
        })) {
      Value copySrc = rewriter.create<linalg::ReshapeOp>(
          reshapeOp.getLoc(), bufferForResult.getType(), adaptor.src(),
          reshapeOp.reassociation());
      rewriter.create<linalg::CopyOp>(reshapeOp.getLoc(), copySrc,
                                      bufferForResult);
    }
    rewriter.replaceOp(reshapeOp, bufferForResult);
    return success();
  }

 private:
  TensorToBufferMap const &resultTensorToBufferMap;
};

struct InitTensorOpConversion
    : public OpConversionPattern<linalg::InitTensorOp> {
  InitTensorOpConversion(MLIRContext *context,
                         TensorToBufferMap const &resultTensorToBufferMap,
                         PatternBenefit benefit = 1)
      : OpConversionPattern<linalg::InitTensorOp>(context, benefit),
        resultTensorToBufferMap(resultTensorToBufferMap) {}

  LogicalResult matchAndRewrite(
      linalg::InitTensorOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Value outputBuffer = resultTensorToBufferMap.lookup(op.result());
    if (!outputBuffer) return failure();
    rewriter.replaceOp(op, outputBuffer);
    return success();
  }

 private:
  TensorToBufferMap const &resultTensorToBufferMap;
};
}  // namespace

//===----------------------------------------------------------------------===//
// tensor.extract op conversion.
//===----------------------------------------------------------------------===//

namespace {

/// A pattern to replace tensor::ExtractOp with LoadOp. Typically, this comes
/// from indirect access in Linalg ops on tensors, eg, TorchIndexSelectOp. The
/// pattern expects other patterns to convert the operand to MemRefType.
struct ExtractElementOpPattern final
    : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!operands[0].getType().isa<MemRefType>()) {
      return op.emitError("expected operands[0] to be a MemRefType");
    }
    tensor::ExtractOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<LoadOp>(op, operands[0], adaptor.indices());
    return success();
  }
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
    if (Value buffer =
            resolveResult(shapeOp.operand(), adaptor.operand(),
                          shapeOp.result(), resultTensorToBufferMap)) {
      rewriter.replaceOp(shapeOp, buffer);
    } else {
      rewriter.replaceOpWithNewOp<Shape::TieShapeOp>(
          shapeOp, getMemrefTypeForTensor(shapeOp.result()), adaptor.operand(),
          adaptor.shape());
    }
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
    if (!matchPattern(loadOp.offset(), m_Zero())) {
      return loadOp.emitError("unhandled non-zero offset");
    }

    // Get the corresponding memref type from the tensor type.
    auto tensorType = loadOp.result().getType().cast<RankedTensorType>();
    auto bindingOp = loadOp.queryBindingOp();
    assert(bindingOp);
    auto bufferType = getTensorBackingBufferType(tensorType, bindingOp.type());

    // Create the placeholder op for the backing buffer. Make sure shape
    // annotation is carried over if exists.
    auto phOp = rewriter.create<IREE::PlaceholderOp>(
        loadOp.getLoc(), bufferType, "interface buffer");
    phOp->setAttr(getBindingAttrName(), loadOp.binding());
    StringRef attrName = getOperandResultNumAttrName();
    if (auto operandResultNumAttr = loadOp->getAttr(attrName))
      phOp->setAttr(attrName, operandResultNumAttr);
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
  HALInterfaceStoreTensorOpEraser(MLIRContext *context,
                                  OutputBufferMap const &outputBufferMap,
                                  PatternBenefit benefit = 1)
      : OpConversionPattern<IREE::HAL::InterfaceStoreTensorOp>(context,
                                                               benefit),
        outputBufferMap(outputBufferMap) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceStoreTensorOp storeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::InterfaceStoreTensorOp::Adaptor adaptor(operands);
    Value operand = adaptor.operand();
    if (!operand.getType().isa<MemRefType>()) {
      return storeOp.emitRemark()
             << "expected replacement operand to be of memref type, got "
             << operand.getType();
    }
    Value outputBuffer = outputBufferMap.lookup(storeOp);
    if (!outputBuffer) return storeOp.emitError() << "undefined output buffer";

    // If we are just storing the buffer back to itself again, we can trivially
    // remove this op. Otherwise, copy the content from the source buffer to the
    // destination buffer.
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
  OutputBufferMap const &outputBufferMap;
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
    IREE::HAL::InterfaceStoreTensorOp op, OutputBufferMap &outputBufferMap,
    TensorToBufferMap &resultTensorToBufferMap, OpBuilder &builder) {
  if (!matchPattern(op.offset(), m_Zero())) {
    return op.emitError("unhandled non-zero offset");
  }

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
  phOp->setAttr(getBindingAttrName(), op.binding());
  StringRef attrName = getOperandResultNumAttrName();
  if (Attribute operandResultNumAttr = op->getAttr(attrName))
    phOp->setAttr(attrName, operandResultNumAttr);
  Value buffer = phOp;
  outputBufferMap[op] = buffer;

  resultTensorToBufferMap.insert(std::make_pair(tensor, buffer));
  while (true) {
    if (auto tieShapeOp = tensor.getDefiningOp<Shape::TieShapeOp>()) {
      if (!tieShapeOp.result().hasOneUse()) break;
      builder.setInsertionPointAfter(tieShapeOp.shape().getDefiningOp());
      auto newTieShapeOp = builder.create<Shape::TieShapeOp>(
          op.getLoc(), buffer.getType(), buffer, tieShapeOp.shape());
      tensor = tieShapeOp.operand();
      buffer = newTieShapeOp.result();
      resultTensorToBufferMap.insert(std::make_pair(tensor, buffer));
      continue;
    }
    if (auto tensorReshapeOp =
            tensor.getDefiningOp<linalg::TensorReshapeOp>()) {
      tensor = tensorReshapeOp.src();
      if (resultTensorToBufferMap.count(tensor)) break;
      auto newReshapeOp = builder.create<linalg::ReshapeOp>(
          op.getLoc(), getMemrefTypeForTensor(tensorReshapeOp.getSrcType()),
          buffer, tensorReshapeOp.reassociation());
      buffer = newReshapeOp.result();
      resultTensorToBufferMap.insert(std::make_pair(tensor, buffer));
      continue;
    }
    if (auto linalgOp = tensor.getDefiningOp<linalg::LinalgOp>()) {
      for (auto en : llvm::enumerate(linalgOp.getOperation()->getResults())) {
        if (en.value() != tensor) continue;
        tensor = linalgOp.getOutputs()[en.index()];
        break;
      }
      resultTensorToBufferMap.insert(std::make_pair(tensor, buffer));
      continue;
    }
    break;
  }
  return success();
}

/// Processes the hal.interface.store.tensor instructions to get buffer views
/// for the inputs/outputs to the dispatch function.
static LogicalResult createAndPropagateBufferUsedForResultTensors(
    FuncOp funcOp, OutputBufferMap &outputBufferMap,
    TensorToBufferMap &resultTensorToBufferMap) {
  OpBuilder builder(funcOp.getBody());
  auto walkResult = funcOp.walk(
      [&](IREE::HAL::InterfaceStoreTensorOp storeTensorOp) -> WalkResult {
        return createAndPropagateBufferUsedForResultTensor(
            storeTensorOp, outputBufferMap, resultTensorToBufferMap, builder);
      });
  return failure(walkResult.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Pass specification.
//===----------------------------------------------------------------------===//

namespace {
struct ConvertHLOToLinalgOnBuffersPass
    : public PassWrapper<ConvertHLOToLinalgOnBuffersPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREEDialect>();
  }

  void runOnFunction() override;
};
}  // namespace

void populateHLOToLinalgOnBuffersConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    TensorToBufferMap const &resultTensorToBufferMap) {
  patterns
      .insert<ConvOpConversion, ConcatenateOpConversion,
              DotOpConversion<DotOperationType::MatrixMatrix, linalg::MatmulOp>,
              DotGeneralOpConversion, InitTensorOpConversion,
              LinalgOpOnTensorConversion<linalg::GenericOp>,
              LinalgOpOnTensorConversion<linalg::IndexedGenericOp>,
              PadOpConversion, ReduceOpConversion, ReduceWindowOpConversion,
              SliceOpConversion, TensorReshapeOpConversion>(
          context, resultTensorToBufferMap);
  // Reduce region operation conversions.
  patterns.insert<ReduceRegionXLAOpConversion<mhlo::AddOp>,
                  ReduceRegionXLAOpConversion<mhlo::MinOp>,
                  ReduceRegionXLAOpConversion<mhlo::MaxOp>,
                  ReduceRegionReturnOpConversion>(context);
}

void ConvertHLOToLinalgOnBuffersPass::runOnFunction() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getFunction();

  // First create buffers for all StoreTensorOps.
  OutputBufferMap outputBufferMap;
  TensorToBufferMap resultTensorToBufferMap;
  if (failed(createAndPropagateBufferUsedForResultTensors(
          funcOp, outputBufferMap, resultTensorToBufferMap))) {
    return signalPassFailure();
  }

  OwningRewritePatternList patterns;
  populateHLOToLinalgOnBuffersConversionPatterns(context, patterns,
                                                 resultTensorToBufferMap);
  patterns.insert<HALInterfaceLoadTensorOpEraser, ShapeOpPattern>(
      context, resultTensorToBufferMap);
  patterns.insert<HALInterfaceStoreTensorOpEraser>(context, outputBufferMap);
  patterns.insert<ExtractElementOpPattern>(context);

  ConversionTarget target(*context);
  // Make sure all XLA HLO ops are converted to Linalg ops after this pass.
  target.addIllegalDialect<mhlo::MhloDialect>();
  // All Linalg ops should operate on buffers. So hal.interface.*.tensor ops
  // should be gone.
  target.addIllegalOp<IREE::HAL::InterfaceLoadTensorOp,
                      IREE::HAL::InterfaceStoreTensorOp, tensor::ExtractOp>();
  target.addDynamicallyLegalOp<Shape::TieShapeOp>(
      [](Shape::TieShapeOp op) -> bool {
        return op.operand().getType().isa<MemRefType>();
      });
  // Also convert away linalg.tensor_reshape.
  target.addIllegalOp<linalg::TensorReshapeOp>();
  target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
      Optional<ConversionTarget::DynamicLegalityCallbackFn>([](Operation *op) {
        // The generated structured Linalg ops should have buffer
        // semantics.
        if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
          return linalgOp.hasBufferSemantics();
        }
        return !isa<linalg::InitTensorOp>(op);
      }));
  // Let the rest fall through.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyFullConversion(getFunction(), target, std::move(patterns)))) {
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
