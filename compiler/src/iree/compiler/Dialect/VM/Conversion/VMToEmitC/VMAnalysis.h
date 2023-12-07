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
#include "mlir/Dialect/EmitC/IR/EmitC.h"

namespace mlir::iree_compiler {

struct VMAnalysis {
public:
  VMAnalysis() = default;
  VMAnalysis(IREE::VM::FuncOp &funcOp) {
    Operation *op = funcOp.getOperation();
    registerAllocation = RegisterAllocation(op);
    valueLiveness = ValueLiveness(op);
    originalFunctionType = funcOp.getFunctionType();
  }
  VMAnalysis(FunctionType functionType) { originalFunctionType = functionType; }

  VMAnalysis(VMAnalysis &&) = default;
  VMAnalysis &operator=(VMAnalysis &&) = default;
  VMAnalysis(const VMAnalysis &) = delete;
  VMAnalysis &operator=(const VMAnalysis &) = delete;

  FunctionType getFunctionType() { return originalFunctionType; }

  int getNumRefRegisters() {
    return registerAllocation.getMaxRefRegisterOrdinal() + 1;
  }

  int getNumRefArguments() {
    assert(originalFunctionType);
    return llvm::count_if(originalFunctionType.getInputs(), [](Type inputType) {
      return inputType.isa<IREE::VM::RefType>();
    });
  }

  int getNumLocalRefs() { return getNumRefRegisters() - getNumRefArguments(); }

  uint16_t getRefRegisterOrdinal(Value ref) {
    assert(ref.getType().isa<IREE::VM::RefType>());
    return registerAllocation.mapToRegister(ref).ordinal();
  }

  bool isMove(Value ref, Operation *op) {
    assert(ref.getType().isa<IREE::VM::RefType>());
    bool lastUse = valueLiveness.isLastValueUse(ref, op);
    return lastUse && false;
  }

  void cacheLocalRef(int64_t ordinal, emitc::ApplyOp &applyOp) {
    assert(!refs.count(ordinal));
    refs[ordinal] = applyOp.getOperation();
  }

  emitc::ApplyOp lookupLocalRef(int64_t ordinal) {
    assert(refs.count(ordinal));
    Operation *op = refs[ordinal];
    return cast<emitc::ApplyOp>(op);
  }

  DenseMap<int64_t, Operation *> &localRefs() { return refs; }

private:
  RegisterAllocation registerAllocation;
  ValueLiveness valueLiveness;
  DenseMap<int64_t, Operation *> refs;
  FunctionType originalFunctionType;
};

using VMAnalysisCache = DenseMap<Operation *, VMAnalysis>;

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMANALYSIS_H_
