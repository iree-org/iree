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

#include "iree/compiler/Dialect/Sequence/IR/SequenceDialect.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Sequence/IR/SequenceOps.h"
#include "iree/compiler/Dialect/Sequence/IR/SequenceTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Sequence {

#include "iree/compiler/Dialect/Sequence/IR/SequenceOpInterface.cpp.inc"

static DialectRegistration<SequenceDialect> sequence_dialect;

SequenceDialect::SequenceDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<SequenceDialect>()) {
  addTypes<SequenceType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Sequence/IR/SequenceOps.cpp.inc"
      >();
}

Type SequenceDialect::parseType(DialectAsmParser& parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  mlir::Type elementType;
  if (parser.parseKeyword("of") || parser.parseLess() ||
      parser.parseType(elementType) || parser.parseGreater()) {
    emitError(loc, "unknown sequence type: ") << spec;
    return Type();
  }
  return SequenceType::getChecked(elementType, loc);
}

void SequenceDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (auto sequenceType = type.dyn_cast<SequenceType>())
    os << "of<" << sequenceType.getTargetType() << ">";
}

}  // namespace Sequence
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
