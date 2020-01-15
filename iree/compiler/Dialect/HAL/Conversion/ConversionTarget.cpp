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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace iree_compiler {

HALConversionTarget::HALConversionTarget(MLIRContext *context,
                                         TypeConverter &typeConverter)
    : ConversionTarget(*context),
      context(*context),
      typeConverter(typeConverter) {
  // Setup the fallback handler such that all ops without explicitly
  // registered patterns will be checked to ensure that they don't use any
  // illegal types.
  setupFallbackTypeLegality();

  // The HAL dialect expects both standard ops and the HAL ops (in case some
  // conversion has already happened).
  addLegalDialect<StandardOpsDialect>();
  addLegalOp<FuncOp, ModuleOp, ModuleTerminatorOp>();
  addLegalDialect<IREE::HAL::HALDialect>();

  // We don't care about the contents of a HAL executable: it may have any kind
  // of dialect and type usage.
  addLegalOp<IREE::HAL::ExecutableOp>();
  markOpRecursivelyLegal<IREE::HAL::ExecutableOp>();

  addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });
  addDynamicallyLegalOp<ConstantOp>(
      [&](ConstantOp op) { return typeConverter.isLegal(op.getType()); });
}

void HALConversionTarget::setupFallbackTypeLegality() {
  // TODO(b/147671560): a way to more cleanly support type fallbacks.
  for (auto *dialect : context.getRegisteredDialects()) {
    addDynamicallyLegalDialect(dialect->getNamespace());
  }
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
    if (auto tensorType =
            srcOperand.getType().template dyn_cast<TensorType>()) {
      if (!tensorType.hasStaticShape()) {
        // Dynamic shapes not yet implemented.
        return srcOp->emitOpError(
            "has a dynamically shaped tensor operand that is not yet "
            "supported");
      }

      // New operand as hal.buffer.
      state.addOperands({dstOperand});

      // Type encoded to a matching primitive data type.
      auto elementType =
          IREE::HAL::getElementTypeValue(tensorType.getElementType());
      if (!elementType) {
        // Unsupported element type.
        return srcOp->emitOpError(
            "requires the element type be mappable to "
            "IREE::HAL::ElementType");
      }
      auto typeOp = rewriter.createOrFold<mlir::ConstantOp>(
          srcOp->getLoc(), rewriter.getIntegerType(32),
          rewriter.getI32IntegerAttr(elementType.getValue()));
      state.addOperands({typeOp});

      // Shape encoded as a variadic list of dimensions.
      // TODO(benvanik): segment_sizes for multiple operands.
      for (int64_t dim : tensorType.getShape()) {
        auto dimOp = rewriter.createOrFold<mlir::ConstantOp>(
            srcOp->getLoc(), rewriter.getIntegerType(32),
            rewriter.getI32IntegerAttr(static_cast<int32_t>(dim)));
        state.addOperands({dimOp});
      }
    } else {
      state.addOperands({dstOperand});
    }
  }
  for (auto resultType : srcOp->getResultTypes()) {
    if (auto tensorType = resultType.template dyn_cast<TensorType>()) {
      if (!tensorType.hasStaticShape()) {
        // Dynamic shapes not yet implemented.
        return srcOp->emitOpError(
            "has a dynamically shaped tensor result that is not yet supported");
      }

      // TODO(benvanik): use type converter multi-type expansion.
      // Tensor -> buffer.
      state.addTypes(typeConverter.convertType(tensorType));
      // Element type.
      state.addTypes(rewriter.getIntegerType(32));
      // Shape encoded as a variadic list of dimensions.
      // TODO(benvanik): segment_sizes for multiple results.
      for (int i = 0; i < tensorType.getShape().size(); ++i) {
        state.addTypes(rewriter.getIntegerType(32));
      }
    } else {
      state.addTypes({resultType});
    }
  }

  auto *dstOp = rewriter.createOperation(state);
  rewriter.replaceOp(srcOp, dstOp->getResults());
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
