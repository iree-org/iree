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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/CodegenUtils/MarkerUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
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

/// Returns a corresponding memref type for the given `tensorType` stored in the
/// given `descriptorType`.
static MemRefType getTensorBackingBufferType(
    TensorType tensorType, IREE::HAL::DescriptorType descriptorType) {
  // Get the memory space from the HAL interface so we can carry that over via
  // memref.
  unsigned memorySpace = mapDescriptorTypeToMemorySpace(descriptorType);

  // Get the corresponding memref type from the tensor type.
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         /*affineMapComposition=*/{}, memorySpace);
}

/// Returns the interface buffer for the given op `operand`. This assumes the
/// given `operand` is a tensor loaded from a HAL inteface buffer.
static Value getBufferForOpOperand(Value operand, OpBuilder &builder) {
  Operation *def = operand.getDefiningOp();

  // There may exist dynamic shape annotation. Penetrate through.
  auto tieShapeOp = dyn_cast_or_null<Shape::TieShapeOp>(def);
  if (tieShapeOp) def = tieShapeOp.operand().getDefiningOp();

  // This operand should be defined by a hal.interface.load.tensor op.
  auto loadOp = dyn_cast_or_null<IREE::HAL::InterfaceLoadTensorOp>(def);
  if (!loadOp || !matchPattern(loadOp.offset(), m_Zero())) return nullptr;
  Location loc = def->getLoc();

  // Get the corresponding memref type from the tensor type.
  auto tensorType = operand.getType().cast<TensorType>();
  auto bindingOp = loadOp.queryBindingOp();
  assert(bindingOp);
  auto bufferType = getTensorBackingBufferType(tensorType, bindingOp.type());

  // Create the placeholder op for the backing buffer. Make sure shape
  // annotation is carried over if exists.
  auto phOp =
      builder.create<IREE::PlaceholderOp>(loc, bufferType, "interface buffer");
  phOp.setAttr("binding", loadOp.binding());
  Value buffer = phOp;
  if (tieShapeOp)
    buffer = builder.create<Shape::TieShapeOp>(loc, buffer, tieShapeOp.shape());
  return buffer;
}

/// Returns the interface buffer for the given op `result`. This assumes the
/// given `result` is a tensor stored to a HAL interface buffer.
static Value getBufferForOpResult(Value result, OpBuilder &builder) {
  if (!result.hasOneUse()) return nullptr;
  Location loc = result.getDefiningOp()->getLoc();
  Operation *use = result.use_begin()->getOwner();

  // There may exist dynamic shape annotation. Penetrate through.
  auto tieShapeOp = dyn_cast<Shape::TieShapeOp>(use);
  if (tieShapeOp) use = tieShapeOp.result().use_begin()->getOwner();

  // This result should be used by a hal.interface.store.tensor op.
  auto storeOp = dyn_cast<IREE::HAL::InterfaceStoreTensorOp>(use);
  if (!storeOp || !matchPattern(storeOp.offset(), m_Zero())) return nullptr;

  // Get the corresponding memref type from the tensor type.
  auto tensorType = result.getType().cast<TensorType>();
  auto bindingOp = storeOp.queryBindingOp();
  assert(bindingOp);
  auto bufferType = getTensorBackingBufferType(tensorType, bindingOp.type());

  // Create the placeholder op for the backing buffer. Make sure shape
  // annotation is carried over if exists.
  auto phOp =
      builder.create<IREE::PlaceholderOp>(loc, bufferType, "interface buffer");
  phOp.setAttr("binding", storeOp.binding());
  Value buffer = phOp;
  if (tieShapeOp)
    buffer = builder.create<Shape::TieShapeOp>(loc, buffer, tieShapeOp.shape());
  return buffer;
}

/// Returns true if the given `operand` is a direct load from an interface
/// tensor.
static bool isDirectlyReadingFromInterfaceTensor(Value operand) {
  Operation *def = operand.getDefiningOp();
  auto tieShapeOp = dyn_cast_or_null<Shape::TieShapeOp>(def);
  if (tieShapeOp) def = tieShapeOp.operand().getDefiningOp();
  return isa_and_nonnull<IREE::HAL::InterfaceLoadTensorOp>(def);
}

/// Returns true if the given `result` is a direct write to an interface tensor.
static bool isDirectlyWritingToInterfaceTensor(Value result) {
  if (!result.hasOneUse()) return false;
  Operation *use = result.use_begin()->getOwner();
  auto tieShapeOp = dyn_cast<Shape::TieShapeOp>(use);
  if (tieShapeOp) use = tieShapeOp.result().use_begin()->getOwner();
  return isa<IREE::HAL::InterfaceStoreTensorOp>(use);
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
/// All derived classes implement an apply method with the following signature:
///
/// ```c++
/// OpTy apply(SrcOpTy op, ArrayRef<Value> args, ArrayRef<Value> results,
///            ConversionPatternRewriter& rewriter) const;
/// ```
///
/// The `op` is the op being converted. `args` contains the buffers to use for
/// as inputs to the converted op, and `results` contains the buffer to use for
/// the outputs of the converted op. The method returns a linalg op on buffers.
template <typename DerivedTy, typename SrcOpTy, typename LinalgOpTy>
struct ConvertToLinalgBufferOp : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Operation *op = srcOp.getOperation();

    // Prepare interface buffers for operands.
    SmallVector<Value, 4> operandBuffers;
    operandBuffers.reserve(operands.size());
    for (auto operand : llvm::enumerate(operands)) {
      // We have special treatment for constant initializers for reduction.
      if (isa<ConstantOp>(operand.value().getDefiningOp())) {
        operandBuffers.push_back(operand.value());
        continue;
      }

      Value operandBuffer = getBufferForOpOperand(operand.value(), rewriter);
      if (!operandBuffer) {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag << "failed to create buffer for operand #" << operand.index();
        });
      }
      operandBuffers.push_back(operandBuffer);
    }

    // Prepare interface buffers for results.
    SmallVector<Value, 1> resultBuffers;
    resultBuffers.reserve(op->getNumResults());
    for (auto result : llvm::enumerate(op->getResults())) {
      Value resultBuffer = getBufferForOpResult(result.value(), rewriter);
      if (!resultBuffer) {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag << "failed to create buffer for result #" << result.index();
        });
      }
      resultBuffers.push_back(resultBuffer);
    }

    // Apply the main conversion logic.
    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    LinalgOpTy dstOp = static_cast<const DerivedTy &>(*this).apply(
        srcOp, operandBuffers, resultBuffers, rewriter);

    if (!dstOp || !dstOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          op, "failed to apply main conversion logic");

    // Ops using this Linalg op's results are expecting tensors. But here we
    // feed them buffers. This is okay because it is hidden as internal state
    // during conversion process. But this relies on collaborating patterns to
    // properly handle ops using the results.
    rewriter.replaceOp(srcOp, resultBuffers);
    return success();
  }
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
                                     xla_hlo::DotOp, LinalgOpTy> {
  using ConvertToLinalgBufferOp<DotOpConversion<opType, LinalgOpTy>,
                                xla_hlo::DotOp,
                                LinalgOpTy>::ConvertToLinalgBufferOp;
  LinalgOpTy apply(xla_hlo::DotOp op, ArrayRef<Value> args,
                   ArrayRef<Value> results,
                   ConversionPatternRewriter &rewriter) const {
    if (getDotOperationType(op) == opType)
      return rewriter.create<LinalgOpTy>(op.getLoc(), args[0], args[1],
                                         results[0]);
    return nullptr;
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// xla_hlo.convolution conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Converts xla_hlo.convolution operation to linalg.conv op.
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
  if (const auto dimensionNumbers = op.dimension_numbers()) {
    const int inputSpatialRank =
        llvm::size(dimensionNumbers.input_spatial_dimensions());
    // The dimensions for input should follow the order of
    // batch_count, spatial_dims..., input_feature_count.
    if (dimensionNumbers.input_batch_dimension().getInt() != 0 ||
        dimensionNumbers.input_feature_dimension().getInt() !=
            (inputSpatialRank + 1))
      return nullptr;

    const int kernelSpatialRank =
        llvm::size(dimensionNumbers.kernel_spatial_dimensions());
    // The dimensions for filter should follow the order of
    // spatial_dims..., input_feature_count, num_output_feature_count.
    if (dimensionNumbers.kernel_input_feature_dimension().getInt() !=
            kernelSpatialRank ||
        dimensionNumbers.kernel_output_feature_dimension().getInt() !=
            (kernelSpatialRank + 1))
      return nullptr;

    const int outputSpatialRank =
        llvm::size(dimensionNumbers.output_spatial_dimensions());
    // The dimensions for output should follow the order of
    // batch_count, spatial_dims.., output_feature_count.
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

  return rewriter.create<linalg::ConvOp>(op.getLoc(), args[1], args[0],
                                         results[0], stridesArg, dilationArg,
                                         padding);
}

//===----------------------------------------------------------------------===//
// xla_hlo.pad conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Converts xla_hlo.pad operation to linalg.indexed_generic op.
// TODO(GH-1604): Lower the pad op to a Linalg named op.
struct PadOpConversion
    : public ConvertToLinalgBufferOp<PadOpConversion, xla_hlo::PadOp,
                                     linalg::IndexedGenericOp> {
  using ConvertToLinalgBufferOp<
      PadOpConversion, xla_hlo::PadOp,
      linalg::IndexedGenericOp>::ConvertToLinalgBufferOp;

  AffineMapAttr getInputIndexingMap(xla_hlo::PadOp op, int rank,
                                    ConversionPatternRewriter &rewriter) const;
  linalg::IndexedGenericOp apply(xla_hlo::PadOp op, ArrayRef<Value> args,
                                 ArrayRef<Value> results,
                                 ConversionPatternRewriter &rewriter) const;
};
}  // namespace

AffineMapAttr PadOpConversion::getInputIndexingMap(
    xla_hlo::PadOp op, int rank, ConversionPatternRewriter &rewriter) const {
  const auto edgePaddingLow = convertDenseIntAttr(op.edge_padding_low());
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0; i < rank; ++i)
    exprs.push_back((rewriter.getAffineDimExpr(i) - edgePaddingLow[i]));
  return AffineMapAttr::get(
      AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext()));
}

linalg::IndexedGenericOp PadOpConversion::apply(
    xla_hlo::PadOp op, ArrayRef<Value> args, ArrayRef<Value> results,
    ConversionPatternRewriter &rewriter) const {
  if (llvm::any_of(op.interior_padding().getValues<IntegerAttr>(),
                   [](auto attr) { return attr.getInt() != 0; })) {
    op.emitError("pad op with non-zero interiror_padding is not supported");
    return nullptr;
  }

  xla_hlo::PadOpOperandAdaptor adaptor(args);
  auto loc = op.getLoc();

  Attribute paddingConstVal = getInitValueAsConst(adaptor.padding_value());
  Value paddingVal =
      paddingConstVal
          ? rewriter.create<ConstantOp>(loc, paddingConstVal).getResult()
          : adaptor.padding_value();

  auto operandType = adaptor.operand().getType().cast<ShapedType>();
  int rank = operandType.getRank();

  SmallVector<Attribute, 2> indexingMaps;
  indexingMaps.emplace_back(getInputIndexingMap(op, rank, rewriter));
  if (!paddingConstVal) {
    indexingMaps.emplace_back(AffineMapAttr::get(
        AffineMap::get(rank, /*symbolCount=*/0, rewriter.getContext())));
  }
  indexingMaps.emplace_back(AffineMapAttr::get(
      AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())));

  SmallVector<Type, 2> resultTypes = {};
  SmallVector<Value, 2> linalgOpArgs = {adaptor.operand()};
  if (!paddingConstVal) linalgOpArgs.push_back(adaptor.padding_value());
  linalgOpArgs.push_back(results[0]);
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
  return linalgOp;
}

//===----------------------------------------------------------------------===//
// xla_hlo.reduce_window conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
struct ReduceWindowOpConversion
    : public ConvertToLinalgBufferOp<
          ReduceWindowOpConversion, xla_hlo::ReduceWindowOp, linalg::LinalgOp> {
  using ConvertToLinalgBufferOp<ReduceWindowOpConversion,
                                xla_hlo::ReduceWindowOp,
                                linalg::LinalgOp>::ConvertToLinalgBufferOp;

  enum class PoolingType {
    kMin,
    kMax,
    kAdd,
  };

  PoolingType getPoolingType(Region &region) const;

  linalg::LinalgOp apply(xla_hlo::ReduceWindowOp op, ArrayRef<Value> operands,
                         ArrayRef<Value> results,
                         ConversionPatternRewriter &rewriter) const;
};
}  // namespace

ReduceWindowOpConversion::PoolingType ReduceWindowOpConversion::getPoolingType(
    Region &region) const {
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

linalg::LinalgOp ReduceWindowOpConversion::apply(
    xla_hlo::ReduceWindowOp op, ArrayRef<Value> operands,
    ArrayRef<Value> results, ConversionPatternRewriter &rewriter) const {
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
                loc, ArrayRef<Type>{}, operands[0], fakeWindowDims.getResult(),
                results[0], stridesArg,
                /*dilations=*/nullptr,
                /*padding=*/nullptr)
            .getOperation());
  };
  linalg::LinalgOp poolingOp;
  PoolingType poolingType = getPoolingType(op.body());
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

  return poolingOp;
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
  SmallVector<Value, 2> linalgOpArgs = {operands[0]};
  if (!initConstVal) linalgOpArgs.push_back(operands[1]);
  linalgOpArgs.push_back(results[0]);
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
  return linalgOp;
}

//===----------------------------------------------------------------------===//
// Linalg op on tensors to linalg op on buffers conversion base class.
//===----------------------------------------------------------------------===//

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

/// Convert linalg.tensor_reshape to a linalg.reshape + linalg.copy. This is
/// only needed when linalg.tensor_reshape ends up in its own dispatch region.
struct SingleReshapeOpInDispatchConversion
    : public ConvertToLinalgBufferOp<SingleReshapeOpInDispatchConversion,
                                     linalg::TensorReshapeOp, linalg::CopyOp> {
  using ConvertToLinalgBufferOp<SingleReshapeOpInDispatchConversion,
                                linalg::TensorReshapeOp,
                                linalg::CopyOp>::ConvertToLinalgBufferOp;
  linalg::CopyOp apply(linalg::TensorReshapeOp op, ArrayRef<Value> args,
                       ArrayRef<Value> results,
                       ConversionPatternRewriter &rewriter) const {
    linalg::ReshapeOp reshapeOp = rewriter.create<linalg::ReshapeOp>(
        op.getLoc(), results[0].getType(), args[0], op.reassociation());

    return rewriter.create<linalg::CopyOp>(op.getLoc(), reshapeOp.getResult(),
                                           results[0]);
  }
};

// The tensor reshape op has copy semantics in general, but if the reshape is of
// an argument tensor, just alias the argument. In the corner case where the
// result of the reshape is written into another argument, then fallback to the
// reshape + copy solution.
struct TensorReshapeOpConversion
    : public OpConversionPattern<linalg::TensorReshapeOp> {
  using OpConversionPattern<linalg::TensorReshapeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      linalg::TensorReshapeOp reshapeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (isDirectlyReadingFromInterfaceTensor(reshapeOp.src()) &&
        !isDirectlyWritingToInterfaceTensor(reshapeOp.result())) {
      RankedTensorType resultType = reshapeOp.getResultType();
      rewriter.replaceOpWithNewOp<linalg::ReshapeOp>(
          reshapeOp,
          MemRefType::get(resultType.getShape(), resultType.getElementType()),
          operands[0], reshapeOp.reassociation());
      return success();
    }
    return failure();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// hal.interface.store.tensor conversion.
//===----------------------------------------------------------------------===//

namespace {
/// Erases hal.interface.store.tensor ops.
///
/// During the convertion to Linalg on buffers, we may generate the following IR
/// sequence temporarily:
///
/// ```mlir
/// %buffer = iree.placeholder for "interface buffer" ...
/// %buffer_shape_tie1 = shapex.tie_shape %buffer ...
/// linalg.generic ... %buffer_shape_tie1 { ... }
/// %buffer_shape_tie2 = shapex.tie_shape %buffer ...
/// hal.interface.store.tensor(%buffer_shape_tie2, %buffer_shape_tie1)
/// ```
///
/// In the above, hal.interface.store.tensor can be erased and it triggers
/// further simpliciation.
struct HALInterfaceStoreTensorOpEraser final
    : public OpConversionPattern<IREE::HAL::InterfaceStoreTensorOp> {
  using OpConversionPattern<
      IREE::HAL::InterfaceStoreTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceStoreTensorOp storeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::InterfaceStoreTensorOpOperandAdaptor adaptor(operands);

    Operation *defOp = adaptor.operand().getDefiningOp();
    Shape::TieShapeOp tieShapeOp;
    // There may exist shape annotation. If so, bypass it.
    if ((tieShapeOp = dyn_cast<Shape::TieShapeOp>(defOp))) {
      defOp = tieShapeOp.operand().getDefiningOp();
    }

    // To erase, this tensor should be loaded from an interface buffer having
    // the same interface binding as the store destination.
    if (auto phOp = dyn_cast<IREE::PlaceholderOp>(defOp)) {
      if (storeOp.binding() != phOp.getAttrOfType<SymbolRefAttr>("binding")) {
        return failure();
      }
    }

    // If the shapex.tie_shape op only have this use, erase it too.
    if (tieShapeOp && tieShapeOp.getResult().hasOneUse()) {
      rewriter.eraseOp(tieShapeOp);
    }
    rewriter.eraseOp(storeOp);

    return success();
  }
};
}  // namespace

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
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns
      .insert<ConvOpConversion,
              DotOpConversion<DotOperationType::MatrixMatrix, linalg::MatmulOp>,
              LinalgOpOnTensorConversion<linalg::GenericOp>, PadOpConversion,
              SingleReshapeOpInDispatchConversion, TensorReshapeOpConversion>(
          context);
  // Reduce Operation conversions.
  patterns.insert<ReduceOpConversion, ReduceWindowOpConversion,
                  ReduceRegionXLAOpConversion<xla_hlo::AddOp>,
                  ReduceRegionXLAOpConversion<xla_hlo::MinOp>,
                  ReduceRegionXLAOpConversion<xla_hlo::MaxOp>,
                  ReduceRegionReturnOpConversion>(context);
}

void ConvertHLOToLinalgOnBuffersPass::runOnFunction() {
  MLIRContext *context = &getContext();

  OwningRewritePatternList patterns;
  populateHLOToLinalgOnBuffersConversionPatterns(context, patterns);
  patterns.insert<HALInterfaceStoreTensorOpEraser>(context);

  ConversionTarget target(*context);
  // Make sure all XLA HLO ops are converted to Linalg ops after this pass.
  target.addIllegalDialect<xla_hlo::XlaHloDialect>();
  // All Linalg ops should operate on buffers. So hal.interface.store.tensor ops
  // should be gone.
  target.addIllegalOp<IREE::HAL::InterfaceStoreTensorOp>();
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
    "iree-hlo-to-linalg-on-buffers",
    "Convert from XLA-HLO ops to Linalg ops on buffers");
}  // namespace iree_compiler
}  // namespace mlir
