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
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

VMLAConversionTarget::VMLAConversionTarget(MLIRContext *context,
                                           TypeConverter &typeConverter)
    : ConversionTarget(*context),
      context(*context),
      typeConverter(typeConverter) {
  // The VMLA dialect expects both standard ops and the VMLA ops (in case some
  // conversion has already happened).
  addLegalOp<ModuleOp, ModuleTerminatorOp>();
  addLegalDialect<StandardOpsDialect>();
  addLegalDialect<IREE::VMLA::VMLADialect>();

  // Allow other ops to pass through so long as their type is valid (not a
  // tensor, basically).
  markUnknownOpDynamicallyLegal();
  addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });
  addDynamicallyLegalOp<ConstantOp>(
      [&](ConstantOp op) { return typeConverter.isLegal(op.getType()); });
}

bool VMLAConversionTarget::isDynamicallyLegal(Operation *op) const {
  // Short-circuit test that bails on the first illegal type.
  const auto isTypeIllegal = [&](Type type) {
    return !typeConverter.isLegal(type);
  };
  return !(llvm::any_of(op->getOperandTypes(), isTypeIllegal) ||
           llvm::any_of(op->getResultTypes(), isTypeIllegal));
}

// static
LogicalResult VMLAConversionTarget::applyDefaultBufferRewrite(
    Operation *srcOp, ArrayRef<Value> operands, StringRef dstOpName,
    TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  // TODO(benvanik): implement rewriting.
  return failure();
}

}  // namespace iree_compiler
}  // namespace mlir
