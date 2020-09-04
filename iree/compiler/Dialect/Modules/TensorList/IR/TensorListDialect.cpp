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

#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListDialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Modules/TensorList/Conversion/ConvertHALToVM.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"
#include "iree/compiler/Dialect/Modules/TensorList/tensorlist.imports.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TensorList {

namespace {

static DialectRegistration<TensorListDialect> registration;

class TensorListToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef getVMImportModule() const override {
    return mlir::parseSourceString(StringRef(tensorlist_imports_create()->data,
                                             tensorlist_imports_create()->size),
                                   getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateTensorListToVMPatterns(getDialect()->getContext(), importSymbols,
                                   patterns, typeConverter);
  }
};

}  // namespace

TensorListDialect::TensorListDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<TensorListDialect>()) {
  addInterfaces<TensorListToVMConversionInterface>();

  addTypes<TensorListType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.cpp.inc"
      >();
}

Type TensorListDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (failed(parser.parseKeyword(&typeName))) return {};
  auto type = llvm::StringSwitch<Type>(typeName)
                  .Case("list", TensorListType::get(getContext()))
                  .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown type: " << typeName;
  }
  return type;
}

void TensorListDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<TensorListType>()) {
    p << "list";
  } else {
    llvm_unreachable("unknown type");
  }
}

}  // namespace TensorList
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
