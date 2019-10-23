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

#include "iree/compiler/IR/Dialect.h"

#include "iree/compiler/IR/ConfigOps.h"
#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/IR/Types.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace iree_compiler {

static DialectRegistration<IREEDialect> iree_dialect;

IREEDialect::IREEDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
#define IREE_ADD_TYPE(NAME, KIND, TYPE) addTypes<TYPE>();
  IREE_TYPE_TABLE(IREE_ADD_TYPE);

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/IR/Ops.cpp.inc"
      >();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/IR/ConfigOps.cpp.inc"
      >();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/IR/StructureOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

#define IREE_TYPE_PARSER(NAME, KIND, TYPE)                            \
  static Type parse##TYPE(IREEDialect const &dialect, StringRef spec, \
                          Location loc) {                             \
    spec.consume_front(NAME);                                         \
    return TYPE::get(dialect.getContext());                           \
  }
IREE_TYPE_TABLE(IREE_TYPE_PARSER);

#define IREE_PARSE_TYPE(NAME, KIND, TYPE) \
  if (spec.startswith(NAME)) {            \
    return parse##TYPE(*this, spec, loc); \
  }
Type IREEDialect::parseType(StringRef spec, Location loc) const {
  IREE_TYPE_TABLE(IREE_PARSE_TYPE);
  emitError(loc, "unknown IREE type: ") << spec;
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

#define IREE_TYPE_PRINTER(NAME, KIND, TYPE) \
  static void print##TYPE(TYPE type, llvm::raw_ostream &os) { os << NAME; }
IREE_TYPE_TABLE(IREE_TYPE_PRINTER);

#define IREE_PRINT_TYPE(NAME, KIND, TYPE) \
  case KIND:                              \
    print##TYPE(type.cast<TYPE>(), os);   \
    return;
void IREEDialect::printType(Type type, llvm::raw_ostream &os) const {
  switch (type.getKind()) {
    IREE_TYPE_TABLE(IREE_PRINT_TYPE);
    default:
      llvm_unreachable("unhandled IREE type");
  }
}

}  // namespace iree_compiler
}  // namespace mlir
