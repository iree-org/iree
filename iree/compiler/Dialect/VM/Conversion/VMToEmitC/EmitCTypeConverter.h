// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/VMAnalysis.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

class EmitCTypeConverter : public mlir::TypeConverter {
 public:
  EmitCTypeConverter();
  FailureOr<std::reference_wrapper<VMAnalysis>> lookupAnalysis(
      mlir::FuncOp &funcOp) {
    return lookupAnalysis(funcOp.getOperation());
  }
  FailureOr<std::reference_wrapper<VMAnalysis>> lookupAnalysis(
      IREE::VM::FuncOp &funcOp) {
    return lookupAnalysis(funcOp.getOperation());
  }
  Optional<Value> materializeRef(Value ref);

  SetVector<Operation *> sourceMaterializations;
  VMAnalysisCache analysisCache;

 private:
  FailureOr<std::reference_wrapper<VMAnalysis>> lookupAnalysis(Operation *op);
};

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCTYPECONVERTER_H_
