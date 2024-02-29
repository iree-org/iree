// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMANALYSIS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMANALYSIS_H_

#include <optional>

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "iree/compiler/Dialect/VM/Utils/CallingConvention.h"
#include "iree/compiler/Dialect/VM/Utils/TypeTable.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

namespace mlir::iree_compiler::IREE::VM {

/// TODO(simon-camp): This struct grew from being a wrapper around the
/// RegisterAllocation and ValueLiveness analyses to also cache other things
/// needed throughout the conversion. This led to hard to locate failures
/// when only part of this struct was correctly initialized. This
/// should be split into multiple structs each with a single responsibility.
struct FuncAnalysis {
  FuncAnalysis() = default;
  FuncAnalysis(bool emitAtEnd) : emitAtEnd(emitAtEnd) {}
  FuncAnalysis(IREE::VM::FuncOp funcOp) {
    Operation *op = funcOp.getOperation();
    registerAllocation = RegisterAllocation(op);
    valueLiveness = ValueLiveness(op);
    originalFunctionType = funcOp.getFunctionType();
    callingConvention = makeCallingConventionString(funcOp).value();
    refs = DenseMap<int64_t, Operation *>{};
  }
  FuncAnalysis(mlir::emitc::FuncOp funcOp) {
    originalFunctionType = funcOp.getFunctionType();
  }
  FuncAnalysis(IREE::VM::ImportOp importOp) {
    originalFunctionType = importOp.getFunctionType();
    callingConvention = makeImportCallingConventionString(importOp).value();
  }
  FuncAnalysis(FunctionType functionType, StringRef cconv) {
    originalFunctionType = functionType;
    callingConvention = cconv.str();
  }
  FuncAnalysis(FuncAnalysis &analysis, StringRef exportName_) {
    originalFunctionType = analysis.getFunctionType();
    callingConvention = analysis.getCallingConvention().str();
    exportName = exportName_.str();
  }

  FuncAnalysis(FuncAnalysis &&) = default;
  FuncAnalysis &operator=(FuncAnalysis &&) = default;
  FuncAnalysis(const FuncAnalysis &) = delete;
  FuncAnalysis &operator=(const FuncAnalysis &) = delete;

  StringRef getCallingConvention() {
    assert(callingConvention.has_value());
    return callingConvention.value();
  }

  StringRef getExportName() {
    assert(exportName.has_value());
    return exportName.value();
  }

  bool isExported() { return exportName.has_value(); }

  bool shouldEmitAtEnd() { return emitAtEnd.value_or(false); }

  FunctionType getFunctionType() {
    assert(originalFunctionType.has_value());
    return originalFunctionType.value();
  }

  int getNumRefRegisters() {
    assert(registerAllocation.has_value());
    return registerAllocation.value().getMaxRefRegisterOrdinal() + 1;
  }

  int getNumRefArguments() {
    assert(originalFunctionType.has_value());
    return llvm::count_if(
        originalFunctionType.value().getInputs(),
        [](Type inputType) { return isa<IREE::VM::RefType>(inputType); });
  }

  int getNumLocalRefs() { return getNumRefRegisters() - getNumRefArguments(); }

  uint16_t getRefRegisterOrdinal(TypedValue<IREE::VM::RefType> ref) {
    assert(registerAllocation.has_value());
    return registerAllocation.value().mapToRegister(ref).ordinal();
  }

  bool isMove(Value ref, Operation *op) {
    assert(isa<IREE::VM::RefType>(ref.getType()));
    assert(valueLiveness.has_value());
    bool lastUse = valueLiveness.value().isLastValueUse(ref, op);
    return lastUse && false;
  }

  void cacheLocalRef(int64_t ordinal, emitc::ApplyOp applyOp) {
    assert(refs.has_value());
    assert(!refs.value().count(ordinal) && "ref was already cached");
    refs.value()[ordinal] = applyOp.getOperation();
  }

  emitc::ApplyOp lookupLocalRef(int64_t ordinal) {
    assert(refs.has_value());
    assert(refs.value().count(ordinal) && "ref not found in cache");
    Operation *op = refs.value()[ordinal];
    return cast<emitc::ApplyOp>(op);
  }

  bool hasLocalRefs() { return refs.has_value(); }

  DenseMap<int64_t, Operation *> &localRefs() {
    assert(refs.has_value());
    return refs.value();
  }

private:
  std::optional<RegisterAllocation> registerAllocation;
  std::optional<ValueLiveness> valueLiveness;
  std::optional<DenseMap<int64_t, Operation *>> refs;
  std::optional<FunctionType> originalFunctionType;
  std::optional<std::string> callingConvention;
  std::optional<std::string> exportName;
  std::optional<bool> emitAtEnd;
};

struct ModuleAnalysis {
  ModuleAnalysis(IREE::VM::ModuleOp module) {
    typeTable = buildTypeTable(module);
    for (auto [index, typeDef] : llvm::enumerate(typeTable)) {
      mapType(typeDef.type, index);
    }

    for (auto func : module.getOps<IREE::VM::FuncOp>()) {
      functions[func.getOperation()] = FuncAnalysis(func);
    }
    for (auto func : module.getOps<mlir::emitc::FuncOp>()) {
      functions[func.getOperation()] = FuncAnalysis(func);
    }

    for (auto func : module.getOps<IREE::VM::FuncOp>()) {
      for (auto &block : func.getBody()) {
        for (Value blockArg : block.getArguments()) {
          if (!isa<IREE::VM::RefType>(blockArg.getType())) {
            continue;
          }
          blockArgMapping[blockArg] = func;
        }
      }
    }
  }

  ModuleAnalysis(ModuleAnalysis &&) = default;
  ModuleAnalysis &operator=(ModuleAnalysis &&) = default;
  ModuleAnalysis(const ModuleAnalysis &) = delete;
  ModuleAnalysis &operator=(const ModuleAnalysis &) = delete;

  void addDummy(mlir::emitc::FuncOp func, bool emitAtEnd) {
    functions[func.getOperation()] = FuncAnalysis(emitAtEnd);
  }

  void addFromExport(mlir::emitc::FuncOp func, IREE::VM::ExportOp exportOp) {
    mlir::emitc::FuncOp funcOp =
        exportOp->getParentOfType<IREE::VM::ModuleOp>()
            .lookupSymbol<mlir::emitc::FuncOp>(exportOp.getFunctionRefAttr());

    auto &funcAnalysis = lookupFunction(funcOp);
    functions[func.getOperation()] =
        FuncAnalysis(funcAnalysis, exportOp.getExportName());
  }

  void addFromImport(mlir::emitc::FuncOp func, IREE::VM::ImportOp import) {
    functions[func.getOperation()] = FuncAnalysis(import);
  }

  void move(mlir::emitc::FuncOp newFunc, IREE::VM::FuncOp oldFunc) {
    auto &analysis = lookupFunction(oldFunc.getOperation());

    functions[newFunc.getOperation()] = std::move(analysis);
    functions.erase(oldFunc.getOperation());
    functionMapping[oldFunc] = newFunc;
  }

  FuncAnalysis &lookupFunction(Operation *op) {
    auto ptr = functions.find(op);
    assert(ptr != functions.end() && "analysis lookup failed");
    return ptr->second;
  }

  void mapType(Type type, size_t index) { typeOrdinalMap[type] = index; }
  std::optional<size_t> lookupType(Type type) const {
    auto ptr = typeOrdinalMap.find(type);
    if (ptr == typeOrdinalMap.end()) {
      return std::nullopt;
    }
    return ptr->second;
  }

  Value lookupRef(Value ref) {
    auto refValue = cast<TypedValue<IREE::VM::RefType>>(ref);

    mlir::emitc::FuncOp funcOp;
    if (auto definingOp = ref.getDefiningOp()) {
      funcOp = definingOp->getParentOfType<mlir::emitc::FuncOp>();
    } else {
      assert(blockArgMapping.contains(ref) && "Mapping does not contain ref");
      funcOp = functionMapping[blockArgMapping[ref]];
    }

    auto &analysis = lookupFunction(funcOp);

    int32_t ordinal = analysis.getRefRegisterOrdinal(refValue);

    auto ctx = funcOp.getContext();

    // Search block arguments
    int refArgCounter = 0;
    for (BlockArgument arg : funcOp.getArguments()) {
      assert(!isa<IREE::VM::RefType>(arg.getType()));

      if (arg.getType() == emitc::PointerType::get(
                               emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
        if (ordinal == refArgCounter++) {
          return arg;
        }
      }
    }

    emitc::ApplyOp applyOp = analysis.lookupLocalRef(ordinal);
    return applyOp.getResult();
  }

  std::vector<TypeDef> typeTable;

private:
  DenseMap<Operation *, FuncAnalysis> functions;
  llvm::DenseMap<Type, int> typeOrdinalMap;
  llvm::DenseMap<Value, IREE::VM::FuncOp> blockArgMapping;
  llvm::DenseMap<IREE::VM::FuncOp, emitc::FuncOp> functionMapping;
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMANALYSIS_H_
