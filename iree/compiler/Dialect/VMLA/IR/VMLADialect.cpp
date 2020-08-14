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

#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"

#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VMLA/Conversion/VMLAToVM/ConvertVMLAToVM.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "iree/compiler/Dialect/VMLA/vmla.imports.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

namespace {

static DialectRegistration<VMLADialect> vmla_dialect;

class VMLAToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef getVMImportModule() const override {
    return mlir::parseSourceString(
        StringRef(vmla_imports_create()->data, vmla_imports_create()->size),
        getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateVMLAToVMPatterns(getDialect()->getContext(), typeConverter,
                             importSymbols, patterns);
  }
};

}  // namespace

VMLADialect::VMLADialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VMLADialect>()) {
  addInterfaces<VMLAToVMConversionInterface>();

  addTypes<BufferType, InterfaceType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type VMLADialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (parser.parseKeyword(&typeName)) return Type();
  auto type = llvm::StringSwitch<Type>(typeName)
                  .Case("buffer", BufferType::get(getContext()))
                  .Case("interface", InterfaceType::get(getContext()))
                  .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown VMLA type: " << typeName;
  }
  return type;
}

void VMLADialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<BufferType>()) {
    p << "buffer";
  } else if (type.isa<InterfaceType>()) {
    p << "interface";
  } else {
    llvm_unreachable("unknown VMLA type");
  }
}

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
