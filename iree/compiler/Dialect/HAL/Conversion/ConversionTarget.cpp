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
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

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
      // Create the buffer view that we'll pass to the function.
      // Note that we expect this to be CSE'd if there are multiple calls using
      // the same buffer.
      IREE::HAL::TensorRewriteAdaptor operand(srcOp->getLoc(), srcOperand,
                                              dstOperand, rewriter);
      state.addOperands({operand.getBufferView()});
    } else {
      // Normal pass-through operand.
      state.addOperands({dstOperand});
    }
  }
  for (auto resultType : srcOp->getResultTypes()) {
    if (auto tensorType = resultType.template dyn_cast<TensorType>()) {
      state.addTypes(IREE::RefPtrType::get(
          IREE::HAL::BufferViewType::get(rewriter.getContext())));
    } else {
      // Normal pass-through result.
      state.addTypes({resultType});
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
    if (auto tensorType = resultType.template dyn_cast<TensorType>()) {
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
