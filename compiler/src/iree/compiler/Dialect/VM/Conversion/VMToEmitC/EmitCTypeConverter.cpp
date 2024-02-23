// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"

#include <functional>

namespace mlir::iree_compiler::IREE::VM {

EmitCTypeConverter::EmitCTypeConverter(ModuleOp module)
    : analysis(ModuleAnalysis(module)) {
  // Return the incoming type in the default case.
  addConversion([](Type type) { return type; });

  addConversion([](emitc::OpaqueType type) { return type; });

  addConversion([](IREE::VM::RefType type) {
    return emitc::PointerType::get(
        emitc::OpaqueType::get(type.getContext(), "iree_vm_ref_t"));
  });

  addTargetMaterialization([this](OpBuilder &builder, emitc::PointerType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    auto input = cast<TypedValue<IREE::VM::RefType>>(inputs[0]);
    return analysis.lookupRef(input);
  });

  // We need a source materialization for refs because after running
  // `applyFullConversion` there would be references to the original
  // IREE::VM::Ref values in unused basic block arguments. As these are unused
  // anyway we create dummy ops which get deleted after the conversion has
  // finished.
  addSourceMaterialization([this](OpBuilder &builder, IREE::VM::RefType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<emitc::PointerType>());

    Type objectType = IREE::VM::OpaqueType::get(builder.getContext());
    Type refType = IREE::VM::RefType::get(objectType);

    auto ctx = builder.getContext();
    auto op = builder.create<emitc::VariableOp>(
        /*location=*/loc,
        /*resultType=*/refType,
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    sourceMaterializations.insert(op.getOperation());

    return op.getResult();
  });
}

Type EmitCTypeConverter::convertTypeAsNonPointer(Type type) const {
  Type convertedType = convertType(type);

  if (auto ptrType = llvm::dyn_cast<emitc::PointerType>(convertedType)) {
    return ptrType.getPointee();
  }

  return convertedType;
}

Type EmitCTypeConverter::convertTypeAsPointer(Type type) const {
  return emitc::PointerType::get(convertTypeAsNonPointer(type));
}

emitc::OpaqueType EmitCTypeConverter::convertTypeAsCType(Type type) const {
  Type convertedType = convertTypeAsNonPointer(type);

  if (auto oType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
    return oType;
  }

  if (auto iType = llvm::dyn_cast<IntegerType>(type)) {
    std::string typeLiteral;
    switch (iType.getWidth()) {
    case 32: {
      typeLiteral = "int32_t";
      break;
    }
    case 64: {
      typeLiteral = "int64_t";
      break;
    }
    default:
      return {};
    }
    return emitc::OpaqueType::get(type.getContext(), typeLiteral);
  }

  if (auto fType = llvm::dyn_cast<FloatType>(type)) {
    std::string typeLiteral;
    switch (fType.getWidth()) {
    case 32: {
      typeLiteral = "float";
      break;
    }
    case 64: {
      typeLiteral = "double";
      break;
    }
    default:
      return {};
    }
    return emitc::OpaqueType::get(type.getContext(), typeLiteral);
  }
  return {};
}

} // namespace mlir::iree_compiler::IREE::VM
