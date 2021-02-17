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

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

#include "iree/compiler/Dialect/Flow/IR/FlowInterfaces.cpp.inc"

namespace {

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
  addInterfaces<FlowFolderInterface>();
  addTypes<DispatchInputType, DispatchOutputType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Flow/IR/FlowOps.cpp.inc"
      >();
  context->getOrLoadDialect("shapex");
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
  if (succeeded(parser.parseOptionalKeyword("dispatch.input"))) {
    return DispatchInputType::parse(parser);
  } else if (succeeded(parser.parseOptionalKeyword("dispatch.output"))) {
    return DispatchOutputType::parse(parser);
  }
  parser.emitError(parser.getCurrentLocation())
      << "unknown Flow type: " << spec;
  return {};
}

void FlowDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (auto inputType = type.dyn_cast<DispatchInputType>()) {
    IREE::Flow::printType(inputType, p);
  } else if (auto outputType = type.dyn_cast<DispatchOutputType>()) {
    IREE::Flow::printType(outputType, p);
  } else {
    llvm_unreachable("unknown Flow type");
  }
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
