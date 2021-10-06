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
    auto originalRef = lookup(ref);
    if (originalRef.hasValue()) {
      assert(originalRef.getValue().getType().isa<IREE::VM::RefType>());
      return registerAllocation.mapToRegister(originalRef.getValue()).ordinal();
    }

    auto ptr = ordinalMapping.find(ref);
    assert(ptr != ordinalMapping.end() &&
           "ref for original block arg not found");
    return ptr->second;
  }

  bool isLastValueUse(Value ref, Operation *op) {
    auto originalRef = lookup(ref);

    if (originalRef.hasValue()) {
      assert(originalRef.getValue().getType().isa<IREE::VM::RefType>());
      return valueLiveness.isLastValueUse(originalRef.getValue(), op);
    }

    auto ptr = lastUseMapping.find({ref, op});
    ref.dump();
    assert(ptr != lastUseMapping.end() &&
           "ref for original block arg not found");
    return ptr->second;
  }

  void mapValue(Value original, Value replacement) {
    assert(original.getType().isa<IREE::VM::RefType>());
    mapping.map(replacement, original);
    return;
  }

  void mapLastUse(Value original, Operation *op, Value replacement) {
    bool lastUse = isLastValueUse(original, op);
    lastUseMapping[{replacement, op}] = lastUse;
  }

  void mapOrdinal(Value original, Value replacement) {
    uint16_t ordinal = getRefRegisterOrdinal(original);
    ordinalMapping[replacement] = ordinal;
  }

 private:
  RegisterAllocation registerAllocation;
  ValueLiveness valueLiveness;
  BlockAndValueMapping mapping;
  DenseMap<Value, uint16_t> ordinalMapping;
  DenseMap<std::pair<Value, Operation *>, bool> lastUseMapping;

  Optional<Value> lookup(Value ref) {
    if (ref.getType().isa<IREE::VM::RefType>()) {
      return ref;
    }

    if (mapping.contains(ref)) {
      Value result = mapping.lookup(ref);

      if (!result.getType().isa<IREE::VM::RefType>()) {
        result.dump();
      }
      assert(result.getType().isa<IREE::VM::RefType>());
      return result;
    }
    return {};
  }
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
