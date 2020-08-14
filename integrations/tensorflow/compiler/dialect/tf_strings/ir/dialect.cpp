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

#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/dialect.h"

#include "integrations/tensorflow/compiler/dialect/tf_strings/conversion/convert_flow_to_hal.h"
#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/ops.h"
#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/types.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace iree_compiler {
namespace tf_strings {

#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/op_interface.cpp.inc"

namespace {
static DialectRegistration<TFStringsDialect> tf_strings_dialect;
}  // namespace

TFStringsDialect::TFStringsDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TFStringsDialect>()) {
  addInterfaces<TfStringsToHALConversionInterface>();
  addTypes<StringType>();

#define GET_OP_LIST
  addOperations<
#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/ops.cpp.inc"
      >();
}

Type TFStringsDialect::parseType(DialectAsmParser& parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec == "string") {
    return StringType::get(getContext());
  }
  emitError(loc, "unknown TFStrings type: ") << spec;
  return Type();
}

void TFStringsDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<tf_strings::StringType>())
    os << "string";
  else
    llvm_unreachable("unhandled string type");
}

bool TFStringsType::classof(Type type) {
  return llvm::isa<TFStringsDialect>(type.getDialect());
}

}  // namespace tf_strings
}  // namespace iree_compiler
}  // namespace mlir
