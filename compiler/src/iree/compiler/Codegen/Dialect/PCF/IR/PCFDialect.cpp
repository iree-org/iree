// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFDialect.h"

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::PCF {

namespace {

// Used to control inlining behavior.
struct PCFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Sure!
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }
};

} // namespace

PCFDialect::PCFDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<PCFDialect>()) {

  registerTypes();
  registerAttributes();
  registerOperations();

  addInterfaces<PCFInlinerInterface>();

  declarePromisedInterface<bufferization::BufferizableOpInterface, GenericOp>();
  declarePromisedInterface<bufferization::BufferizableOpInterface,
                           WriteSliceOp>();
}

} // namespace mlir::iree_compiler::IREE::PCF
