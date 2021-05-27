// Copyright 2021 Google LLC
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

#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"

#include "iree/compiler/Dialect/Modules/VMVX/Conversion/VMVXToVM/ConvertVMVXToVM.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Dialect/Modules/VMVX/vmvx.imports.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

namespace {

class VMVXToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef parseVMImportModule() const override {
    return mlir::parseSourceString(StringRef(iree_vmvx_imports_create()->data,
                                             iree_vmvx_imports_create()->size),
                                   getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateVMVXToVMPatterns(getDialect()->getContext(), typeConverter,
                             importSymbols, patterns);
  }
};

}  // namespace

VMVXDialect::VMVXDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VMVXDialect>()) {
  addInterfaces<VMVXToVMConversionInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.cpp.inc"
      >();
}

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
