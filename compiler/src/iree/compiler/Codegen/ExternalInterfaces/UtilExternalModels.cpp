// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/UtilExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler::IREE::Codegen {

namespace {

//===----------------------------------------------------------------------===//
// ValueBoundsOpInterface
//===----------------------------------------------------------------------===//

struct LoadFromBufferOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          LoadFromBufferOpInterface, IREE::Codegen::LoadFromBufferOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto loadOp = cast<IREE::Codegen::LoadFromBufferOp>(op);
    assert(value == loadOp.getResult() && "invalid value");
    cstr.bound(value)[dim] == cstr.getExpr(loadOp.getBuffer(), dim);
  }
};

} // namespace

void registerUtilExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context,
                            IREE::Codegen::IREECodegenDialect *dialect) {
    IREE::Codegen::LoadFromBufferOp::attachInterface<LoadFromBufferOpInterface>(
        *context);
  });
}

} // namespace mlir::iree_compiler::IREE::Codegen
