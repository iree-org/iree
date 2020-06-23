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

#include "iree/compiler/Dialect/VMLA/Conversion/ConversionTarget.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/VMLA/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATraits.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

using Shape::buildOrFindRankedShapeForValue;

VMLAConversionTarget::VMLAConversionTarget(MLIRContext *context,
                                           TypeConverter &typeConverter)
    : ConversionTarget(*context), typeConverter(typeConverter) {
  // The VMLA dialect expects both standard ops and the VMLA ops (in case some
  // conversion has already happened).
  addLegalOp<ModuleOp, ModuleTerminatorOp>();
  addLegalDialect<IREE::VMLA::VMLADialect>();
  // Pseudo-ops are illegal.
  // If we end up with a lot of these, consider using an "is pseudo" trait.
  addIllegalOp<IREE::VMLA::BatchMatMulPseudoOp>();

  // Allow other ops to pass through so long as their type is valid (not a
  // tensor, basically).
  markUnknownOpDynamicallyLegal();
  addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  addDynamicallyLegalOp<ConstantOp>(
      [&](ConstantOp op) { return typeConverter.isLegal(op.getType()); });
  addLegalOp<ReturnOp>();
}

bool VMLAConversionTarget::isDynamicallyLegal(Operation *op) const {
  // Short-circuit test that bails on the first illegal type.
  const auto isTypeIllegal = [&](Type type) {
    return !typeConverter.isLegal(type);
  };
  return !(llvm::any_of(op->getOperandTypes(), isTypeIllegal) ||
           llvm::any_of(op->getResultTypes(), isTypeIllegal));
}

static Attribute convertAttribute(Attribute srcAttribute) {
  auto *context = srcAttribute.getContext();
  Type attrType = srcAttribute.getType();
  auto elementsAttr = srcAttribute.dyn_cast<ElementsAttr>();
  auto tensorType = attrType.dyn_cast<RankedTensorType>();
  auto indexType = IndexType::get(context);
  auto i64Type = IntegerType::get(64, context);
  // Detect and convert index and i64 tensor attributes to i32 since these
  // invariably must be imported as some kind of VM constant, and the VM is
  // 32bit only.
  // TODO(laurenzo): Remove the i64 match once the HLO ops are defined in terms
  // of index for shape components (vs i64).
  if (elementsAttr && tensorType &&
      (tensorType.getElementType() == i64Type ||
       tensorType.getElementType() == indexType)) {
    auto i32Type = IntegerType::get(32, context);
    using func_type = APInt(const APInt &);
    return elementsAttr.mapValues(
        i32Type, llvm::function_ref<func_type>([](const APInt &in) -> APInt {
          int64_t inValue = in.getSExtValue();
          return APInt(32, inValue, true);
        }));
  }

  return srcAttribute;
}

// static
LogicalResult VMLAConversionTarget::applyDefaultBufferRewrite(
    Operation *srcOp, ArrayRef<Value> operands, VMLAOpSemantics semantics,
    StringRef dstOpName, TypeConverter &typeConverter,
    ConversionPatternRewriter &rewriter) {
  OperationState state{srcOp->getLoc(), dstOpName};
  for (auto srcAttrPair : srcOp->getAttrs()) {
    state.addAttribute(srcAttrPair.first, convertAttribute(srcAttrPair.second));
  }

  auto *dstOperation = state.name.getAbstractOperation();
  auto *opInterface = dstOperation->getInterface<IREE::VMLA::VMLAOp>();

  // Allow the op to get at any of the type information it requires. For
  // example, if the op may later need to know the type of the elements in a
  // type-erased buffer it can stash the original tensor type as an attribute.
  if (opInterface) {
    opInterface->extractTypeAttributes(
        state, llvm::to_vector<4>(srcOp->getOperandTypes()),
        llvm::to_vector<4>(srcOp->getResultTypes()));
  }

  // Until MLIR supports unsigned types we need to sidechannel this to the
  // VMLA->VM conversion that really needs to know.
  switch (semantics) {
    default:
      break;
    case VMLAOpSemantics::kForceUnsigned:
      state.addAttribute("force_unsigned", UnitAttr::get(srcOp->getContext()));
      break;
  }

  // Add all input operands.
  for (auto srcDstOperand : llvm::zip(srcOp->getOperands(), operands)) {
    auto srcOperand = std::get<0>(srcDstOperand);
    auto dstOperand = std::get<1>(srcDstOperand);
    if (auto tensorType =
            srcOperand.getType().template dyn_cast<TensorType>()) {
      // Some ops also require shape information.
      state.addOperands({dstOperand});
      if (dstOperation->hasTrait<OpTrait::IREE::VMLA::IncludeShapes>()) {
        Value operandShape = getTensorShape(srcOp->getLoc(), srcOperand,
                                            typeConverter, rewriter);
        if (!operandShape) {
          return srcOp->emitError() << "failed to get operand tensor shape";
        }
        state.addOperands({operandShape});
      }
    } else {
      // Normal pass-through operand.
      state.addOperands({dstOperand});
    }
  }

  // Allocate output buffers for tensors returned by the op. We'll append these
  // to the operands in order (as is convention here).
  SmallVector<Value, 4> allocatedBuffers;
  for (auto srcResult : srcOp->getResults()) {
    if (auto tensorType = srcResult.getType().template dyn_cast<TensorType>()) {
      auto dstBuffer = allocateOutputBuffer(srcOp->getLoc(), srcResult,
                                            typeConverter, rewriter);
      if (!dstBuffer) {
        return srcOp->emitError()
               << "failed to allocate output buffer for tensor result";
      }
      state.addOperands({dstBuffer});
      allocatedBuffers.push_back(dstBuffer);
      if (dstOperation->hasTrait<OpTrait::IREE::VMLA::IncludeShapes>()) {
        Value resultShape =
            getTensorShape(srcOp->getLoc(), srcResult, typeConverter, rewriter);
        if (!resultShape) {
          return srcOp->emitError() << "failed to get operand tensor shape";
        }
        state.addOperands({resultShape});
      }
    } else {
      // Normal pass-through result.
      state.addTypes({srcResult.getType()});
    }
  }

  // Rebuild the result list and replace the op ensuring that all original op
  // results are represented in order even if we changed them to out params.
  auto *dstOp = rewriter.createOperation(state);
  auto dstResults = llvm::to_vector<4>(dstOp->getResults());
  SmallVector<Value, 4> resultValues;
  for (auto resultType : srcOp->getResultTypes()) {
    if (resultType.template isa<TensorType>()) {
      resultValues.push_back(allocatedBuffers.front());
      allocatedBuffers.erase(allocatedBuffers.begin());
    } else {
      resultValues.push_back(dstResults.front());
      dstResults.erase(dstResults.begin());
    }
  }
  rewriter.replaceOp(srcOp, resultValues);
  return success();
}

// static
Value VMLAConversionTarget::getTensorShape(
    Location loc, Value originalValue, TypeConverter &typeConverter,
    ConversionPatternRewriter &rewriter) {
  return buildOrFindRankedShapeForValue(loc, originalValue,
                                        rewriter.getIndexType(), rewriter);
}

// static
Value VMLAConversionTarget::getBufferOffset(
    Location loc, Value tensorValue, Value indicesValue,
    TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  auto indicesType = indicesValue.getType().cast<ShapedType>();
  SmallVector<Value, 4> indices(indicesType.getNumElements());
  for (int i = 0; i < indicesType.getNumElements(); ++i) {
    auto extractIndex = rewriter.createOrFold<mlir::ConstantIndexOp>(loc, i);
    indices[i] = rewriter.createOrFold<mlir::ExtractElementOp>(
        loc, indicesValue, ValueRange{extractIndex});
  }
  return getBufferOffset(loc, tensorValue, indices, typeConverter, rewriter);
}

// static
Value VMLAConversionTarget::getBufferOffset(
    Location loc, Value tensorValue, ValueRange indices,
    TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  // Element type byte length as the base.
  auto tensorType = tensorValue.getType().cast<ShapedType>();
  auto elementType = tensorType.getElementType();
  auto elementSize = rewriter.createOrFold<mlir::ConstantIndexOp>(
      loc, VMLATypeConverter::getRoundedElementByteWidth(elementType));

  auto shape = getTensorShape(loc, tensorValue, typeConverter, rewriter);
  if (!shape) {
    return nullptr;
  }
  Value offset = rewriter.createOrFold<mlir::ConstantIndexOp>(loc, 0);
  for (int i = 0; i < tensorType.getRank(); ++i) {
    auto axisOffset = indices[i];
    for (int j = i + 1; j < tensorType.getRank(); ++j) {
      auto dim = rewriter.createOrFold<Shape::RankedDimOp>(
          loc, rewriter.getIntegerType(32), shape, j);
      axisOffset = rewriter.createOrFold<mlir::MulIOp>(loc, axisOffset, dim);
    }
    offset = rewriter.createOrFold<mlir::AddIOp>(loc, offset, axisOffset);
  }
  return rewriter.createOrFold<mlir::MulIOp>(loc, offset, elementSize);
}

// static
Value VMLAConversionTarget::getBufferLength(
    Location loc, Value tensorValue, TypeConverter &typeConverter,
    ConversionPatternRewriter &rewriter) {
  // Element type byte length as the base.
  auto tensorType = tensorValue.getType().cast<ShapedType>();
  auto elementType = tensorType.getElementType();
  auto elementSize = rewriter.createOrFold<mlir::ConstantIndexOp>(
      loc, VMLATypeConverter::getRoundedElementByteWidth(elementType));

  auto shape = getTensorShape(loc, tensorValue, typeConverter, rewriter);
  if (!shape) return nullptr;
  auto dims =
      rewriter.create<Shape::RankedDimsOp>(loc, rewriter.getIndexType(), shape);
  Value length = elementSize;
  for (auto dim : dims.getResults()) {
    length = rewriter.createOrFold<mlir::MulIOp>(loc, length, dim);
  }
  return length;
}

// static
Value VMLAConversionTarget::allocateOutputBuffer(
    Location loc, Value originalValue, TypeConverter &typeConverter,
    ConversionPatternRewriter &rewriter) {
  // Compute the required buffer size. Since we are always dense (right now)
  // this is just normal x*y*z*...
  Value byteLength =
      getBufferLength(loc, originalValue, typeConverter, rewriter);
  if (!byteLength) {
    return nullptr;
  }

  // Allocate the buffer of the required size.
  // The caller can then use the buffer instead of the original SSA value.
  return rewriter.createOrFold<IREE::VMLA::BufferAllocOp>(
      loc, IREE::VMLA::BufferType::get(rewriter.getContext()), byteLength);
}

}  // namespace iree_compiler
}  // namespace mlir
