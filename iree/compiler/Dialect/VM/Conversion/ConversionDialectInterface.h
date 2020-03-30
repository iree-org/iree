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

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONDIALECTINTERFACE_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONDIALECTINTERFACE_H_

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// An interface for dialects to expose VM conversion functionality.
// The VM conversion pass will query used dialects via this interface to find
// import definitions and conversion patterns that map from the source dialect
// to the VM dialect.
class VMConversionDialectInterface
    : public DialectInterface::Base<VMConversionDialectInterface> {
 public:
  VMConversionDialectInterface(Dialect *dialect) : Base(dialect) {}

  // Returns a module containing one or more vm.modules with vm.import ops.
  // These modules will be merged into the module being compiled to provide
  // import definitions to the conversion and lowering process.
  virtual OwningModuleRef getVMImportModule() const = 0;

  // Populates |patterns| with rewrites that convert from the implementation
  // dialect to the VM dialect. Many of these can just be default conversions
  // via the VMImportOpConversion class.
  //
  // |importSymbols| contains all vm.imports that have been queried from all
  // used dialects, not just this dialect.
  virtual void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const = 0;

  // Walks all child attributes defined within a custom dialect attribute.
  virtual void walkAttributeStorage(
      Attribute attr,
      const function_ref<void(Attribute elementAttr)> &fn) const {}
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONDIALECTINTERFACE_H_
