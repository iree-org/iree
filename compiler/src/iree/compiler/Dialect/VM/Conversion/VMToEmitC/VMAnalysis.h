// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMANALYSIS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMANALYSIS_H_

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "iree/compiler/Dialect/VM/Utils/CallingConvention.h"
#include "iree/compiler/Dialect/VM/Utils/TypeTable.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::iree_compiler::IREE::VM {

struct FuncAnalysis {
public:
  FuncAnalysis() = default;
  FuncAnalysis(IREE::VM::FuncOp funcOp) {
    Operation *op = funcOp.getOperation();
    registerAllocation = RegisterAllocation(op);
    valueLiveness = ValueLiveness(op);
    originalFunctionType = funcOp.getFunctionType();
    callingConvention = makeCallingConventionString(funcOp).value();
  }
  FuncAnalysis(IREE::VM::ImportOp importOp) {
    originalFunctionType = importOp.getFunctionType();
    callingConvention = makeImportCallingConventionString(importOp).value();
  }
  FuncAnalysis(FunctionType functionType, StringRef cconv) {
    originalFunctionType = functionType;
    callingConvention = cconv.str();
  }
  FuncAnalysis(FuncAnalysis &analysis) {
    originalFunctionType = analysis.getFunctionType();
    callingConvention = analysis.getCallingConvention().str();
  }

  FuncAnalysis(FuncAnalysis &&) = default;
  FuncAnalysis &operator=(FuncAnalysis &&) = default;
  FuncAnalysis(const FuncAnalysis &) = delete;
  FuncAnalysis &operator=(const FuncAnalysis &) = delete;

  StringRef getCallingConvention() { return callingConvention; }

  FunctionType getFunctionType() { return originalFunctionType; }

  int getNumRefRegisters() {
    return registerAllocation.getMaxRefRegisterOrdinal() + 1;
  }

  int getNumRefArguments() {
    assert(originalFunctionType);
    return llvm::count_if(originalFunctionType.getInputs(), [](Type inputType) {
      return isa<IREE::VM::RefType>(inputType);
    });
  }

  int getNumLocalRefs() { return getNumRefRegisters() - getNumRefArguments(); }

  uint16_t getRefRegisterOrdinal(TypedValue<IREE::VM::RefType> ref) {
    return registerAllocation.mapToRegister(ref).ordinal();
  }

  bool isMove(Value ref, Operation *op) {
    assert(isa<IREE::VM::RefType>(ref.getType()));
    bool lastUse = valueLiveness.isLastValueUse(ref, op);
    return lastUse && false;
  }

  void cacheLocalRef(int64_t ordinal, emitc::ApplyOp applyOp) {
    assert(!refs.count(ordinal) && "ref was already cached");
    refs[ordinal] = applyOp.getOperation();
  }

  emitc::ApplyOp lookupLocalRef(int64_t ordinal) {
    assert(refs.count(ordinal) && "ref not found in cache");
    Operation *op = refs[ordinal];
    return cast<emitc::ApplyOp>(op);
  }

  DenseMap<int64_t, Operation *> &localRefs() { return refs; }

private:
  RegisterAllocation registerAllocation;
  ValueLiveness valueLiveness;
  DenseMap<int64_t, Operation *> refs;
  FunctionType originalFunctionType;
  std::string callingConvention;
};

struct ModuleAnalysis {
  ModuleAnalysis(IREE::VM::ModuleOp module) {
    typeTable = buildTypeTable(module);
    for (auto func : module.getOps<IREE::VM::FuncOp>()) {
      functions[func.getOperation()] = FuncAnalysis(func);
    }
  }

  ModuleAnalysis(ModuleAnalysis &&) = default;
  ModuleAnalysis &operator=(ModuleAnalysis &&) = default;
  ModuleAnalysis(const ModuleAnalysis &) = delete;
  ModuleAnalysis &operator=(const ModuleAnalysis &) = delete;

  void addDummy(mlir::func::FuncOp func) {
    functions[func.getOperation()] = FuncAnalysis();
  }

  void addFromExport(mlir::func::FuncOp func, IREE::VM::ExportOp exportOp) {
    mlir::func::FuncOp funcOp =
        exportOp->getParentOfType<IREE::VM::ModuleOp>()
            .lookupSymbol<mlir::func::FuncOp>(exportOp.getFunctionRefAttr());

    auto &funcAnalysis = lookupFunction(funcOp);
    functions[func.getOperation()] = FuncAnalysis(funcAnalysis);
  }

  void addFromImport(mlir::func::FuncOp func, IREE::VM::ImportOp import) {
    functions[func.getOperation()] = FuncAnalysis(import);
  }

  void move(mlir::func::FuncOp newFunc, IREE::VM::FuncOp oldFunc) {
    auto &analysis = lookupFunction(oldFunc.getOperation());

    functions[newFunc.getOperation()] = std::move(analysis);
    functions.erase(oldFunc.getOperation());
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

    mlir::func::FuncOp funcOp;
    if (auto definingOp = ref.getDefiningOp()) {
      funcOp = definingOp->getParentOfType<mlir::func::FuncOp>();
    } else {
      Operation *op = llvm::cast<BlockArgument>(ref).getOwner()->getParentOp();
      funcOp = cast<mlir::func::FuncOp>(op);
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
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMANALYSIS_H_
