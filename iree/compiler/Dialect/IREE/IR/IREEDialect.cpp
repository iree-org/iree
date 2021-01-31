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

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {

IREEDialect::IREEDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<IREEDialect>()) {
  addTypes<IREE::ByteBufferType, IREE::ListType, IREE::MutableByteBufferType,
           IREE::PtrType>();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/IREE/IR/IREEOps.cpp.inc"
      >();
}

Type IREEDialect::parseType(DialectAsmParser& parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec.consume_front("ptr")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed ptr type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    auto variableType = mlir::parseType(spec, getContext());
    if (!variableType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid ptr object type specification: '"
          << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    return IREE::PtrType::getChecked(variableType, loc);
  } else if (spec == "byte_buffer") {
    return IREE::ByteBufferType::get(getContext());
  } else if (spec == "mutable_byte_buffer") {
    return IREE::MutableByteBufferType::get(getContext());
  } else if (spec.consume_front("list")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed list type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    auto elementType = mlir::parseType(spec, getContext());
    if (!elementType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid list element type specification: '"
          << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    return IREE::ListType::getChecked(elementType, loc);
  }
  emitError(loc, "unknown IREE type: ") << spec;
  return Type();
}

void IREEDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (auto ptrType = type.dyn_cast<IREE::PtrType>()) {
    os << "ptr<" << ptrType.getTargetType() << ">";
  } else if (type.isa<IREE::ByteBufferType>()) {
    os << "byte_buffer";
  } else if (type.isa<IREE::MutableByteBufferType>()) {
    os << "mutable_byte_buffer";
  } else if (auto listType = type.dyn_cast<IREE::ListType>()) {
    os << "list<" << listType.getElementType() << ">";
  } else {
    llvm_unreachable("unhandled IREE type");
  }
}

}  // namespace iree_compiler
}  // namespace mlir
