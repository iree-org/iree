// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"

#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

HALConversionTarget::HALConversionTarget(MLIRContext *context,
                                         TypeConverter &typeConverter)
    : ConversionTarget(*context), typeConverter(typeConverter) {
  // Setup the fallback handler such that all ops without explicitly
  // registered patterns will be checked to ensure that they don't use any
  // illegal types.
  markUnknownOpDynamicallyLegal();

  // The HAL dialect expects both standard ops and the HAL ops (in case some
  // conversion has already happened).
  addLegalDialect<StandardOpsDialect>();
  addLegalOp<ModuleOp, ModuleTerminatorOp>();
  addLegalDialect<IREE::HAL::HALDialect>();

  // There are a variety of patterns which convert std.dim and std.rank ops
  // to corresponding HAL ops. All should be eliminated.
  addIllegalOp<DimOp>();
  addIllegalOp<RankOp>();

  // Metadata ops are dynamically legal if their types are legal.
  addDynamicallyLegalOp<Shape::TieShapeOp>([&](Shape::TieShapeOp op) {
    return typeConverter.isLegal(op.result().getType());
  });

  // We don't care about the contents of a HAL executable: it may have any kind
  // of dialect and type usage.
  addLegalOp<IREE::HAL::ExecutableOp>();
  markOpRecursivelyLegal<IREE::HAL::ExecutableOp>();

  addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  addDynamicallyLegalOp<ConstantOp>(
      [&](ConstantOp op) { return typeConverter.isLegal(op.getType()); });
}

bool HALConversionTarget::isDynamicallyLegal(Operation *op) const {
  // Short-circuit test that bails on the first illegal type.
  const auto isTypeIllegal = [&](Type type) {
    return !typeConverter.isLegal(type);
  };
  return !(llvm::any_of(op->getOperandTypes(), isTypeIllegal) ||
           llvm::any_of(op->getResultTypes(), isTypeIllegal));
}

// static
LogicalResult HALConversionTarget::applyDefaultBufferRewrite(
    Operation *srcOp, ArrayRef<Value> operands, StringRef dstOpName,
    TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  OperationState state{srcOp->getLoc(), dstOpName};
  state.addAttributes(srcOp->getAttrs());

  for (auto srcDstOperand : llvm::zip(srcOp->getOperands(), operands)) {
    auto srcOperand = std::get<0>(srcDstOperand);
    auto dstOperand = std::get<1>(srcDstOperand);
    if (HALTypeConverter::ShouldConvertToHalBuffer(srcOperand.getType())) {
      // Create the buffer view that we'll pass to the function.
      // Note that we expect this to be CSE'd if there are multiple calls
      // using the same buffer.
      auto operand = IREE::HAL::TensorRewriteAdaptor::getChecked(
          srcOp->getLoc(), srcOperand, dstOperand, rewriter);
      if (!operand.hasValue()) {
        return srcOp->emitOpError() << "unable to create adaptor for operand";
      }
      auto bufferView = operand->getBufferView();
      if (!bufferView) {
        return srcOp->emitOpError() << "unable to get buffer view for operand";
      }
      state.addOperands({bufferView});
    } else {
      // Normal pass-through operand.
      state.addOperands({dstOperand});
    }
  }
  for (auto resultType : srcOp->getResultTypes()) {
    if (HALTypeConverter::ShouldConvertToHalBuffer(resultType)) {
      state.addTypes(IREE::HAL::BufferViewType::get(rewriter.getContext()));
    } else {
      // Normal pass-through result.
      if (failed(typeConverter.convertType(resultType, state.types))) {
        return failure();
      }
    }
  }

  auto *dstOp = rewriter.createOperation(state);

  // Now unpack any of the buffer views we may have returned.
  SmallVector<Value, 4> results;
  for (auto resultTypeValue :
       llvm::zip(srcOp->getResultTypes(), dstOp->getResults())) {
    Type resultType;
    Value resultValue;
    std::tie(resultType, resultValue) = resultTypeValue;
    if (HALTypeConverter::ShouldConvertToHalBuffer(resultType)) {
      results.push_back(rewriter.createOrFold<IREE::HAL::BufferViewBufferOp>(
          srcOp->getLoc(), resultValue));
    } else {
      results.push_back(resultValue);
    }
  }

  rewriter.replaceOp(srcOp, results);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
