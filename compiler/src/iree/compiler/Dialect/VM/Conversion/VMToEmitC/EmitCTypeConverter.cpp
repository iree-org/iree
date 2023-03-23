// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"

#include <functional>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

EmitCTypeConverter::EmitCTypeConverter() {
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
    assert(inputs[0].getType().isa<IREE::VM::RefType>());
    Value ref = inputs[0];
    std::optional<Value> result = materializeRef(ref);
    return result.has_value() ? result.value() : Value{};
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

Type EmitCTypeConverter::convertTypeAsNonPointer(Type type) {
  Type convertedType = convertType(type);

  if (auto ptrType = convertedType.dyn_cast<emitc::PointerType>()) {
    return ptrType.getPointee();
  }

  return convertedType;
}

Type EmitCTypeConverter::convertTypeAsPointer(Type type) {
  return emitc::PointerType::get(convertTypeAsNonPointer(type));
}

emitc::OpaqueType EmitCTypeConverter::convertTypeAsCType(Type type) {
  Type convertedType = convertTypeAsNonPointer(type);

  if (auto oType = convertedType.dyn_cast<emitc::OpaqueType>()) {
    return oType;
  }

  if (auto iType = type.dyn_cast<IntegerType>()) {
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

  if (auto fType = type.dyn_cast<FloatType>()) {
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

FailureOr<std::reference_wrapper<VMAnalysis>>
EmitCTypeConverter::lookupAnalysis(Operation *op) {
  auto ptr = analysisCache.find(op);
  if (ptr == analysisCache.end()) {
    op->emitError() << "parent func op not found in cache.";
    return failure();
  }
  return std::ref(ptr->second);
}

// TODO(simon-camp): Make this a target materialization and cleanup the call
// sites in the conversion.
std::optional<Value> EmitCTypeConverter::materializeRef(Value ref) {
  assert(ref.getType().isa<IREE::VM::RefType>());

  mlir::func::FuncOp funcOp;
  if (auto definingOp = ref.getDefiningOp()) {
    funcOp = definingOp->getParentOfType<mlir::func::FuncOp>();
  } else {
    Operation *op = ref.cast<BlockArgument>().getOwner()->getParentOp();
    funcOp = cast<mlir::func::FuncOp>(op);
  }

  auto vmAnalysis = lookupAnalysis(funcOp);
  if (failed(vmAnalysis)) {
    funcOp.emitError() << "parent func op not found in cache.";
    return std::nullopt;
  }

  int32_t ordinal = vmAnalysis.value().get().getRefRegisterOrdinal(ref);

  auto ctx = funcOp.getContext();

  // Search block arguments
  int refArgCounter = 0;
  for (BlockArgument arg : funcOp.getArguments()) {
    assert(!arg.getType().isa<IREE::VM::RefType>());

    if (arg.getType() ==
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
      if (ordinal == refArgCounter++) {
        return arg;
      }
    }
  }

  emitc::ApplyOp applyOp = vmAnalysis.value().get().lookupLocalRef(ordinal);
  return applyOp.getResult();
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
