// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONTARGET_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONTARGET_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// A conversion target for the VM dialect that handles some boilerplate for
// nested module conversion.
// Conversions targeting the VM dialect should always use this.
class VMConversionTarget : public ConversionTarget {
public:
  // Ensures that a module has double-nesting to allow for module conversion.
  // If the module is already nested then this is a no-op.
  // Returns a pair of (outer module, inner module).
  //
  // Example:
  //  module { func.func @foo() { ... } }
  // ->
  //  module attributes {vm.toplevel} { module { func.func @foo() { ... } } }
  static std::pair<mlir::ModuleOp, mlir::ModuleOp>
  nestModuleForConversion(mlir::ModuleOp outerModuleOp);

  // Returns whether this is the outer module as setup via
  // nestModuleForConversion. Use for patterns which need to distinguish.
  static bool isTopLevelModule(mlir::ModuleOp moduleOp);

  VMConversionTarget(MLIRContext *context);
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONTARGET_H_
