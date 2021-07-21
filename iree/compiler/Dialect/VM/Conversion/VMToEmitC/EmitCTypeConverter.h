// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_TYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_TYPECONVERTER_H_

#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

class EmitCTypeConverter : public mlir::TypeConverter {
 public:
  EmitCTypeConverter() {
    // Return the incoming type in the default case.
    addConversion([](Type type) { return type; });

    addConversion([](emitc::OpaqueType type) { return type; });

    addConversion([](IREE::VM::RefType type) {
      return emitc::OpaqueType::get(type.getContext(), "iree_vm_ref_t*");
    });
  }
};

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_TYPECONVERTER_H_
