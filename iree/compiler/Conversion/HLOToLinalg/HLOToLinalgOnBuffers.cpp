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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hlo-to-linalg-on-buffers"

namespace mlir {
namespace iree_compiler {

using OutputBufferMap = DenseMap<Operation *, Value>;

// -----------------------------------------------------------------------------
// Utility functions.
// -----------------------------------------------------------------------------

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
      // TODO(hanchung): Remove the buffer allocation once every lowering moves
      // to tensors world. The current logic only works for linalg::LinalgOp, so
      // we still need to allocate buffers for some ops, e.g., mhlo.conv, etc.
      if (!resultBuffer) {
        if (auto shapedType = result.value().getType().dyn_cast<ShapedType>()) {
          if (shapedType.hasStaticShape()) {
            resultBuffer = rewriter.create<memref::AllocOp>(
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
// linalg.pad_tensor conversion patterns and utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Converts linalg.pad_tensor operation to fill + subview + copy ops.
struct PadTensorOpConversion
    : public ConvertToLinalgBufferOp<PadTensorOpConversion,
                                     linalg::PadTensorOp> {
  using ConvertToLinalgBufferOp<PadTensorOpConversion,
                                linalg::PadTensorOp>::ConvertToLinalgBufferOp;

  LogicalResult apply(linalg::PadTensorOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const;
};
}  // namespace

LogicalResult PadTensorOpConversion::apply(
    linalg::PadTensorOp op, ArrayRef<Value> inputBuffers,
    ArrayRef<Value> resultBuffers, ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto yieldOp = cast<linalg::YieldOp>(op.region().begin()->getTerminator());
  rewriter.create<linalg::FillOp>(loc, resultBuffers[0], yieldOp.values()[0]);
  SmallVector<Value, 4> sizes, strides;
  int rank = op.getSourceType().getRank();
  for (int i = 0; i < rank; ++i) {
    sizes.push_back(rewriter.create<memref::DimOp>(loc, inputBuffers[0], i));
    strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
  }
  auto subViewOp = rewriter.create<memref::SubViewOp>(loc, resultBuffers[0],
                                                      op.low(), sizes, strides);
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
// subtensor conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Extracts int64_t values from the assumed ArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(llvm::map_range(
      attr.cast<ArrayAttr>(),
      [](Attribute a) -> int64_t { return a.cast<IntegerAttr>().getInt(); }));
}

/// Converts subtensor operation to subview + linalg.copy
struct SubTensorOpConversion
    : public ConvertToLinalgBufferOp<SubTensorOpConversion, SubTensorOp> {
  using ConvertToLinalgBufferOp<SubTensorOpConversion,
                                SubTensorOp>::ConvertToLinalgBufferOp;

  LogicalResult apply(SubTensorOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto subViewOp = rewriter.create<memref::SubViewOp>(
        loc, inputBuffers[0], op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    rewriter.create<linalg::CopyOp>(loc, subViewOp, resultBuffers[0]);
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// subtensor_insert conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts subtensor_insert operation to subview + linalg.copy.
/// Note: this assumes dest and result are the same buffer.
struct SubTensorInsertOpConversion
    : public ConvertToLinalgBufferOp<SubTensorInsertOpConversion,
                                     SubTensorInsertOp> {
  using ConvertToLinalgBufferOp<SubTensorInsertOpConversion,
                                SubTensorInsertOp>::ConvertToLinalgBufferOp;

  LogicalResult apply(SubTensorInsertOp op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto subViewOp = rewriter.create<memref::SubViewOp>(
        loc, resultBuffers[0], op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    if (auto cstOp = inputBuffers[0].getDefiningOp<ConstantOp>()) {
      auto inputConstAttr = cstOp.valueAttr().cast<DenseElementsAttr>();
      if (!inputConstAttr.isSplat()) {
        return rewriter.notifyMatchFailure(
            op, "non-splat constant is not supported");
      }
      Value cstVal =
          rewriter.create<ConstantOp>(loc, inputConstAttr.getSplatValue());
      rewriter.create<linalg::FillOp>(loc, subViewOp, cstVal);
    } else {
      rewriter.create<linalg::CopyOp>(loc, inputBuffers[0], subViewOp);
    }
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// Linalg op on tensors to linalg op on buffers conversion base class.
//===----------------------------------------------------------------------===//

namespace {

struct FillOpOnTensorConversion
    : public ConvertToLinalgBufferOp<FillOpOnTensorConversion, linalg::FillOp> {
  using ConvertToLinalgBufferOp<FillOpOnTensorConversion,
                                linalg::FillOp>::ConvertToLinalgBufferOp;
  LogicalResult apply(linalg::FillOp fillOp, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const {
    if (!fillOp.hasTensorSemantics()) return failure();
    rewriter.create<linalg::FillOp>(fillOp.getLoc(), resultBuffers[0],
                                    fillOp.value());
    return success();
  }
};

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

/// Converts a linalg named op on tensors to linalg named op on buffers.
template <typename LinalgOpTy>
struct NamedOpConversion
    : public ConvertToLinalgBufferOp<NamedOpConversion<LinalgOpTy>,
                                     LinalgOpTy> {
  using ConvertToLinalgBufferOp<NamedOpConversion<LinalgOpTy>,
                                LinalgOpTy>::ConvertToLinalgBufferOp;
  LogicalResult apply(LinalgOpTy op, ArrayRef<Value> inputBuffers,
                      ArrayRef<Value> resultBuffers,
                      ConversionPatternRewriter &rewriter) const {
    if (!op.hasTensorSemantics()) return failure();
    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    SmallVector<Value, 8> newOperands;
    newOperands.append(inputBuffers.begin(),
                       inputBuffers.end() - op.getNumResults());
    newOperands.append(resultBuffers.begin(), resultBuffers.end());
    auto otherOperands = linalgOp.getAssumedNonShapedOperands();
    newOperands.append(otherOperands.begin(), otherOperands.end());
    Location loc = op.getLoc();
    linalgOp.clone(rewriter, loc, /*resultTypes=*/TypeRange{}, newOperands);
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
    if (!outputBuffer) {
      // If the outputBuffer does not exist, this is a shape-only operand.
      // Allocate a temp buffer and it will get deleted after lowering to loops.
      RankedTensorType type = op.getType();
      auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
      rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType);
    } else {
      rewriter.replaceOp(op, outputBuffer);
    }
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
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, operands[0],
                                                adaptor.indices());
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
static Value createBufferForResultTensor(IREE::HAL::InterfaceStoreTensorOp op,
                                         OpBuilder &builder) {
  if (!matchPattern(op.offset(), m_Zero())) {
    op.emitError("unhandled non-zero offset");
    return nullptr;
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
  return phOp.getResult();
}

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
static LogicalResult propagateBufferUsedForResultTensor(
    Value tensor, Value buffer, TensorToBufferMap &resultTensorToBufferMap,
    OpBuilder &builder, Location loc) {
  resultTensorToBufferMap.insert(std::make_pair(tensor, buffer));
  while (true) {
    if (auto tieShapeOp = tensor.getDefiningOp<Shape::TieShapeOp>()) {
      if (!tieShapeOp.result().hasOneUse()) break;
      builder.setInsertionPointAfter(tieShapeOp.shape().getDefiningOp());
      auto newTieShapeOp = builder.create<Shape::TieShapeOp>(
          loc, buffer.getType(), buffer, tieShapeOp.shape());
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
          loc, getMemrefTypeForTensor(tensorReshapeOp.getSrcType()), buffer,
          tensorReshapeOp.reassociation());
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
    if (auto subTensorInsertOp = tensor.getDefiningOp<SubTensorInsertOp>()) {
      tensor = subTensorInsertOp.dest();
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
  if (funcOp.getBlocks().size() != 1) {
    return funcOp.emitError("expected a single block");
  }

  // Walks in a reverse way, because we create placeholders for output buffers
  // and temp buffers, and propagate them to their defining ops.
  OpBuilder builder(funcOp.getBody());
  auto &block = funcOp.front();
  for (auto op = block.rbegin(); op != block.rend(); op++) {
    if (auto storeTensorOp = dyn_cast<IREE::HAL::InterfaceStoreTensorOp>(*op)) {
      Value tensor = storeTensorOp.operand();
      Value buffer = createBufferForResultTensor(storeTensorOp, builder);
      outputBufferMap[storeTensorOp] = buffer;
      if (failed(propagateBufferUsedForResultTensor(tensor, buffer,
                                                    resultTensorToBufferMap,
                                                    builder, op->getLoc()))) {
        return failure();
      }
      continue;
    }

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(*op)) {
      for (auto result : llvm::enumerate(op->getResults())) {
        Value resultBuffer = resultTensorToBufferMap.lookup(result.value());
        if (resultBuffer) continue;
        if (auto shapedType = result.value().getType().dyn_cast<ShapedType>()) {
          if (shapedType.hasStaticShape()) {
            resultBuffer = builder.create<memref::AllocOp>(
                op->getLoc(), getMemrefTypeForTensor(shapedType));
          }
        }
        if (!resultBuffer) {
          return op->emitError("failed to create buffer for result #")
                 << result.index();
        }
        if (failed(propagateBufferUsedForResultTensor(
                result.value(), resultBuffer, resultTensorToBufferMap, builder,
                op->getLoc()))) {
          return failure();
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns.
//===----------------------------------------------------------------------===//

// TODO(hanchung): Revisit the pattern, this seems no longer needed because the
// reshape ops are folded in tensors world.
// Folds linalg.reshape op that directly reshaping an iree.placeholder op into
// the iree.placeholder op itself.
class FoldReshapeIntoPlaceholder final
    : public OpRewritePattern<linalg::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto placeholderOp = reshapeOp.src().getDefiningOp<IREE::PlaceholderOp>();
    if (!placeholderOp) return failure();
    rewriter.replaceOpWithNewOp<IREE::PlaceholderOp>(
        reshapeOp, reshapeOp.getResultType(), ValueRange(),
        placeholderOp->getAttrs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass specification.
//===----------------------------------------------------------------------===//

namespace {
struct ConvertHLOToLinalgOnBuffersPass
    : public PassWrapper<ConvertHLOToLinalgOnBuffersPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, IREEDialect, memref::MemRefDialect>();
  }

  void runOnFunction() override;
};
}  // namespace

void populateHLOToLinalgOnBuffersConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    TensorToBufferMap const &resultTensorToBufferMap) {
  patterns.insert<
      // clang-format off
      FillOpOnTensorConversion,
      InitTensorOpConversion,
      LinalgOpOnTensorConversion<linalg::GenericOp>,
      LinalgOpOnTensorConversion<linalg::IndexedGenericOp>,
      NamedOpConversion<linalg::ConvInputNWCFilterWCFOp>,
      NamedOpConversion<linalg::ConvInputNHWCFilterHWCFOp>,
      NamedOpConversion<linalg::ConvInputNDHWCFilterDHWCFOp>,
      NamedOpConversion<linalg::DepthwiseConvInputNHWCFilterHWCOp>,
      NamedOpConversion<linalg::DepthwiseConvInputNHWCFilterHWCFOp>,
      NamedOpConversion<linalg::MatmulOp>,
      NamedOpConversion<linalg::MatmulI8I8I32Op>,
      NamedOpConversion<linalg::MatmulI16I16I32Op>,
      NamedOpConversion<linalg::MatmulI32I32I32Op>,
      NamedOpConversion<linalg::BatchMatmulOp>,
      NamedOpConversion<linalg::PoolingNHWCMaxFOp>,
      NamedOpConversion<linalg::PoolingNHWCMinFOp>,
      NamedOpConversion<linalg::PoolingNHWCSumFOp>,
      PadTensorOpConversion,
      SubTensorOpConversion,
      SubTensorInsertOpConversion,
      TensorReshapeOpConversion
      // clang-format on
      >(context, resultTensorToBufferMap);
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

  OwningRewritePatternList patterns(&getContext());
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
                      IREE::HAL::InterfaceStoreTensorOp, SubTensorOp,
                      SubTensorInsertOp, tensor::ExtractOp>();
  target.addDynamicallyLegalOp<Shape::TieShapeOp>(
      [](Shape::TieShapeOp op) -> bool {
        return op.operand().getType().isa<MemRefType>();
      });
  // Also convert away linalg.tensor_reshape and linalg.pad_tensor.
  target.addIllegalOp<linalg::TensorReshapeOp, linalg::PadTensorOp>();
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

  // Perform additional canonicalizations.
  {
    OwningRewritePatternList foldingPatterns(&getContext());
    foldingPatterns.insert<FoldReshapeIntoPlaceholder>(context);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(foldingPatterns));
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
