// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/VMAnalysis.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

void populateVMToEmitCPatterns(ConversionTarget &conversionTarget,
                               IREE::VM::EmitCTypeConverter &typeConverter,
                               RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::VM {
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>> createConvertVMToEmitCPass();
} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_
