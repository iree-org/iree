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

namespace mlir {
namespace iree_compiler {

struct VMAnalysis {
 public:
  VMAnalysis() = default;
  VMAnalysis(IREE::VM::FuncOp &funcOp) {
    Operation *op = funcOp.getOperation();
    registerAllocation = RegisterAllocation(op);
    valueLiveness = ValueLiveness(op);
    originalFunctionType = funcOp.getType();
  }

  VMAnalysis(VMAnalysis &&) = default;
  VMAnalysis &operator=(VMAnalysis &&) = default;
  VMAnalysis(const VMAnalysis &) = delete;
  VMAnalysis &operator=(const VMAnalysis &) = delete;

  int getNumRefRegisters() {
    return registerAllocation.getMaxRefRegisterOrdinal() + 1;
  }

  uint16_t getRefRegisterOrdinal(Value ref) {
    assert(ref.getType().isa<IREE::VM::RefType>());
    return registerAllocation.mapToRegister(ref).ordinal();
  }

  bool isLastValueUse(Value ref, Operation *op) {
    assert(ref.getType().isa<IREE::VM::RefType>());
    return valueLiveness.isLastValueUse(ref, op);
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
  size_t numRefArguments;
  FunctionType originalFunctionType;

 private:
  RegisterAllocation registerAllocation;
  ValueLiveness valueLiveness;
  DenseMap<int64_t, Operation *> refs;
};

using VMAnalysisCache = DenseMap<Operation *, VMAnalysis>;

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMANALYSIS_H_
