// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONDIALECTINTERFACE_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONDIALECTINTERFACE_H_

#include <mutex>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

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
  mlir::ModuleOp getVMImportModule() const {
    std::call_once(importParseFlag,
                   [&]() { importModuleRef = parseVMImportModule(); });
    return importModuleRef.get();
  }

  // Populates |patterns| with rewrites that convert from the implementation
  // dialect to the VM dialect. Many of these can just be default conversions
  // via the VMImportOpConversion class.
  //
  // |importSymbols| contains all vm.imports that have been queried from all
  // used dialects, not just this dialect.
  virtual void
  populateVMConversionPatterns(SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter) const = 0;

  // Walks all child attributes defined within a custom dialect attribute;
  // returns false on unknown attributes.
  virtual LogicalResult walkAttributeStorage(
      Attribute attr,
      const function_ref<void(Attribute elementAttr)> &fn) const {
    return success();
  }

protected:
  // Parses the vm.import module to be cached by the caller.
  virtual OwningOpRef<mlir::ModuleOp> parseVMImportModule() const = 0;

private:
  mutable std::once_flag importParseFlag;
  mutable OwningOpRef<mlir::ModuleOp> importModuleRef;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONDIALECTINTERFACE_H_
