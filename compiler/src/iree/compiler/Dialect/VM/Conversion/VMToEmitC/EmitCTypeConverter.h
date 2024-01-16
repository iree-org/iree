// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/VMAnalysis.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::VM {

class EmitCTypeConverter : public mlir::TypeConverter {
public:
  EmitCTypeConverter(ModuleOp module);

  // This is the same as convertType, but returns `iree_vm_ref_t` rather than a
  // pointer to it for `vm.ref` types.
  Type convertTypeAsNonPointer(Type type) const;
  Type convertTypeAsPointer(Type type) const;
  emitc::OpaqueType convertTypeAsCType(Type type) const;
  std::optional<std::string> convertTypeToStringLiteral(Type type) const;

  SetVector<Operation *> sourceMaterializations;
  mutable ModuleAnalysis analysis;
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_
