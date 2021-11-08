// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_

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

    // We need a source materialization for refs because after running
    // `applyFullConversion` there would be references to the original
    // IREE::VM::Ref values in unused basic block arguments. As these are unused
    // anyway we create dummy ops which get deleted after the conversion has
    // finished.
    addSourceMaterialization([this](OpBuilder &builder, IREE::VM::RefType type,
                                    ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      Value input = inputs[0];
      assert(input.getType().isa<emitc::OpaqueType>());

      Type objectType = IREE::VM::OpaqueType::get(builder.getContext());
      Type refType = IREE::VM::RefType::get(objectType);

      auto ctx = builder.getContext();
      auto op = builder.create<emitc::ConstantOp>(
          /*location=*/loc,
          /*resultType=*/refType,
          /*value=*/emitc::OpaqueAttr::get(ctx, ""));

      sourceMaterializations.insert(op.getOperation());

      return op.getResult();
    });
  }

  SetVector<Operation *> sourceMaterializations;
};

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_
