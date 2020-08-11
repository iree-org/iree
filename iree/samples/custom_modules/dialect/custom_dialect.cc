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

#include "iree/samples/custom_modules/dialect/custom_dialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/samples/custom_modules/dialect/conversion_patterns.h"
#include "iree/samples/custom_modules/dialect/custom.imports.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Custom {

namespace {

static DialectRegistration<CustomDialect> custom_dialect;

// Exposes conversion patterns that transition tensors to buffers during the
// Flow->HAL dialect lowering. This is only required if the dialect has ops that
// use tensor types.
class CustomToHALConversionInterface : public HALConversionDialectInterface {
 public:
  using HALConversionDialectInterface::HALConversionDialectInterface;

  void setupConversionTarget(ConversionTarget &target,
                             OwningRewritePatternList &patterns,
                             TypeConverter &typeConverter) const override {
    populateCustomToHALPatterns(getDialect()->getContext(), patterns,
                                typeConverter);
  }
};

// Exposes the import module and conversion patterns used to convert custom
// ops to their vm.import counterparts.
class CustomToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef getVMImportModule() const override {
    return mlir::parseSourceString(
        StringRef(custom_imports_create()->data, custom_imports_create()->size),
        getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateCustomToVMPatterns(getDialect()->getContext(), importSymbols,
                               patterns, typeConverter);
  }
};

}  // namespace

CustomDialect::CustomDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<CustomDialect>()) {
  addInterfaces<CustomToHALConversionInterface,
                CustomToVMConversionInterface>();

  addTypes<MessageType>();

#define GET_OP_LIST
  addOperations<
#include "iree/samples/custom_modules/dialect/custom_ops.cc.inc"
      >();
}

Type CustomDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (failed(parser.parseKeyword(&typeName))) return {};
  auto type = llvm::StringSwitch<Type>(typeName)
                  .Case("message", MessageType::get(getContext()))
                  .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown type: " << typeName;
  }
  return type;
}

void CustomDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<MessageType>()) {
    p << "message";
  } else {
    llvm_unreachable("unknown type");
  }
}

#define GET_OP_CLASSES
#include "iree/samples/custom_modules/dialect/custom_ops.cc.inc"

}  // namespace Custom
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
