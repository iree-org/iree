// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

#include "iree/compiler/Dialect/Flow/IR/FlowInterfaces.cpp.inc"  // IWYU pragma: keep

namespace {

// Used to control inlining behavior.
struct FlowInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Sure!
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Sure!
    return true;
  }
};

struct FlowFolderInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(Region *region) const override {
    // TODO(benvanik): redirect constants to the region scope when small.
    return false;
  }
};

}  // namespace

FlowDialect::FlowDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<FlowDialect>()) {
  addInterfaces<FlowInlinerInterface, FlowFolderInterface>();
  addTypes<DispatchTensorType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Flow/IR/FlowOps.cpp.inc"
      >();
  context->getOrLoadDialect("shapex");
  context->getOrLoadDialect<tensor::TensorDialect>();
}

Operation *FlowDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (ConstantOp::isBuildableWith(value, type))
    return builder.create<ConstantOp>(loc, type, value);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type FlowDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (succeeded(parser.parseOptionalKeyword("dispatch.tensor"))) {
    return DispatchTensorType::parse(parser);
  }
  parser.emitError(parser.getCurrentLocation())
      << "unknown Flow type: " << spec;
  return {};
}

void FlowDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (auto inputType = type.dyn_cast<DispatchTensorType>()) {
    IREE::Flow::printType(inputType, p);
  } else {
    llvm_unreachable("unknown Flow type");
  }
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
