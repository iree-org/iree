// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VM_PASSES
#define IREE_DIALECT_VM_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

def ConversionPass :
    Pass<"iree-vm-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from various dialects (standard, HAL, etc) to the VM dialect.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::math::MathDialect",
    "::mlir::iree_compiler::IREE::Util::UtilDialect",
    "::mlir::iree_compiler::IREE::VM::VMDialect"
  ];
  let options = [
    Option<"indexBits", "index-bits", "int",
           /*default=*/"64",
           "Target size of `index` when converted to an integer in bits.">,
    Option<"f32Extension", "f32-extension", "bool",
           /*default=*/"true",
           "Whether the f32 extension is enabled in the target VM.">,
    Option<"f64Extension", "f64-extension", "bool",
           /*default=*/"true",
           "Whether the f64 extension is enabled in the target VM.">,
    Option<"truncateUnsupportedFloats", "truncate-unsupported-floats", "bool",
           /*default=*/"true",
           "Truncate f64 types to f32 when the f64 extension is not enabled.">,
    Option<"optimizeForStackSize", "optimize-for-stack-size", "bool",
           /*default=*/"false",
           "Prefer optimizations that reduce VM stack usage over performance.">,
  ];
}

//===----------------------------------------------------------------------===//
// Module layout
//===----------------------------------------------------------------------===//

def ReifyRodataTablesPass :
    Pass<"iree-vm-reify-rodata-tables", "IREE::VM::ModuleOp"> {
  let summary = "Reifies and pads `vm.rodata.table.inline` ops.";
  let description = [{
    Converts `vm.rodata.table.inline` ops into two `vm.rodata.inline` ops: one
    for the flat data and the other for a newly constructed table for the
    element subviews.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::VM::VMDialect"
  ];
}

def HoistInlinedRodataPass :
    Pass<"iree-vm-hoist-inlined-rodata", "IREE::VM::ModuleOp"> {
  let summary = "Hoists inline vm.rodata.inline values to module-level constant storage.";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::VM::VMDialect"
  ];
}

def DeduplicateRodataPass :
    Pass<"iree-vm-deduplicate-rodata", "IREE::VM::ModuleOp"> {
  let summary = "Deduplicates vm.rodata ops in the module.";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::VM::VMDialect"
  ];
}

def ResolveRodataLoadsPass :
    Pass<"iree-vm-resolve-rodata-loads", "IREE::VM::ModuleOp"> {
  let summary = "Resolves global loads of rodata ops to direct rodata references.";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::VM::VMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Module analysis and ordinal assignment
//===----------------------------------------------------------------------===//

def GlobalInitializationPass :
    Pass<"iree-vm-global-initialization", "IREE::VM::ModuleOp"> {
  let summary = "Creates module-level global init/deinit functions.";
  let description = [{
    Gathers all module-level global init/deinit functions into a single
    location such that the runtime can init/deinit everything at once. This
    hides the initialization order rules from the runtime behind singular
    invocations (`__init` and `__deinit`) and prevents the need for the dynamic
    linker to need initialization order metadata.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::VM::VMDialect"
  ];
}

def OrdinalAllocationPass :
    Pass<"iree-vm-ordinal-allocation", "IREE::VM::ModuleOp"> {
  let summary = "Assigns module-unique ordinals to function/global/etc symbols within the module.";
  let description = [{
    Assigns per-category ordinals to module-level symbols in the module.
    Each ordinal is unique per-category and ordinals are contiguous starting
    from zero.
  }];
}

//===----------------------------------------------------------------------===//
// Optimization passes
//===----------------------------------------------------------------------===//

def DropEmptyModuleInitializersPass :
    Pass<"iree-vm-drop-empty-module-initializers", "IREE::VM::ModuleOp"> {
  let summary = "Drops __init/__deinit functions that have no ops.";
}

def DropUnusedCallsPass :
    Pass<"iree-vm-drop-unused-calls", "IREE::VM::ModuleOp"> {
  let summary = "Drops vm.call ops that have no side effects and are unused.";
}

def SinkDefiningOpsPass :
    Pass<"iree-vm-sink-defining-ops", "IREE::VM::ModuleOp"> {
  let summary = "Sinks defining ops with few uses to their use-sites.";
  let description = [{
    Sinks defining ops with few uses to their use-sites to reduce the total
    number of live registers at the cost of additional storage requirements.
  }];
}

#endif  // IREE_DIALECT_VM_PASSES
