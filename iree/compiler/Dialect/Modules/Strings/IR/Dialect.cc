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

#include "iree/compiler/Dialect/Modules/Strings/IR/Dialect.h"

#include "iree/compiler/Dialect/Modules/Strings/Conversion/StringsToVM.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.h"
#include "iree/compiler/Dialect/Modules/Strings/strings.imports.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Strings {

namespace {

static DialectRegistration<StringsDialect> strings_dialect;

class StringsToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef getVMImportModule() const override {
    return mlir::parseSourceString(StringRef(strings_imports_create()->data,
                                             strings_imports_create()->size),
                                   getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateStringsToVMPatterns(getDialect()->getContext(), importSymbols,
                                patterns, typeConverter);
  }
};

}  // namespace

StringsDialect::StringsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<StringsDialect>()) {
  addInterfaces<StringsToVMConversionInterface>();

  addTypes<StringType, StringTensorType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.cc.inc"
      >();
}

Type StringsDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (failed(parser.parseKeyword(&typeName))) return {};
  auto type = llvm::StringSwitch<Type>(typeName)
                  .Case("string", StringType::get(getContext()))
                  .Case("string_tensor", StringTensorType::get(getContext()))
                  .Default(nullptr);

  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown type: " << typeName;
  }
  return type;
}

void StringsDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<StringType>()) {
    p << "string";
  } else if (type.isa<StringTensorType>()) {
    p << "string_tensor";
  } else {
    llvm_unreachable("unknown type");
  }
}

}  // namespace Strings
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
