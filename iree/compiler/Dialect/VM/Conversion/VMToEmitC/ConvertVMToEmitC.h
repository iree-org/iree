// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

struct VMAnalysis {
 public:
  VMAnalysis(RegisterAllocation &&registerAllocation,
             ValueLiveness &&valueLiveness)
      : registerAllocation(std::move(registerAllocation)),
        valueLiveness(std::move(valueLiveness)) {}

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

  void cacheLocalRef(int64_t ordinal, Operation *op) {
    assert(!localRefs.count(ordinal));
    localRefs[ordinal] = op;
  }

  Operation *lookupLocalRef(int64_t ordinal) {
    assert(localRefs.count(ordinal));
    return localRefs[ordinal];
  }

 private:
  RegisterAllocation registerAllocation;
  ValueLiveness valueLiveness;
  DenseMap<int64_t, Operation *> localRefs;
};

using VMAnalysisCache = DenseMap<Operation *, VMAnalysis>;

void populateVMToEmitCPatterns(MLIRContext *context,
                               IREE::VM::EmitCTypeConverter &typeConverter,
                               OwningRewritePatternList &patterns,
                               VMAnalysisCache &vmAnalysisCache);

namespace IREE {
namespace VM {

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>> createConvertVMToEmitCPass();

}  // namespace VM
}  // namespace IREE

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_
