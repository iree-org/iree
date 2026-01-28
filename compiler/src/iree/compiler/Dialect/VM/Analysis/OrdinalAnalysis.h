// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_ORDINALANALYSIS_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_ORDINALANALYSIS_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::iree_compiler::IREE::VM {

// Computes ordinal assignments for module-level symbols.
//
// Each ordinal is unique per-category and ordinals are contiguous starting
// from zero. Categories include:
// - Internal functions (vm.func)
// - Import functions (vm.import)
// - Export functions (vm.export)
// - Rodata segments (vm.rodata)
// - Global refs (vm.global.ref)
// - Global bytes (byte offset for primitive globals)
//
// This analysis is computed on-demand when ordinals are needed for
// serialization, avoiding the need to store ordinals as attributes on ops.
class OrdinalAnalysis {
public:
  // Summary counts of module-level symbols.
  struct OrdinalCounts {
    int32_t importFuncs = 0;
    int32_t exportFuncs = 0;
    int32_t internalFuncs = 0;
    int32_t globalBytes = 0;
    int32_t globalRefs = 0;
    int32_t rodatas = 0;
    int32_t rwdatas = 0; // Currently unused, reserved.
  };

  OrdinalAnalysis() = default;
  explicit OrdinalAnalysis(IREE::VM::ModuleOp moduleOp);

  OrdinalAnalysis(OrdinalAnalysis &&) = default;
  OrdinalAnalysis &operator=(OrdinalAnalysis &&) = default;
  OrdinalAnalysis(const OrdinalAnalysis &) = delete;
  OrdinalAnalysis &operator=(const OrdinalAnalysis &) = delete;

  // Returns the ordinal counts for the module.
  const OrdinalCounts &getCounts() const { return counts_; }

  // Returns the ordinal for a vm.func op.
  int64_t getOrdinal(IREE::VM::FuncOp op) const;

  // Returns the ordinal for a vm.export op.
  int64_t getOrdinal(IREE::VM::ExportOp op) const;

  // Returns the ordinal for a vm.import op.
  int64_t getOrdinal(IREE::VM::ImportOp op) const;

  // Returns the ordinal for a vm.rodata op.
  int64_t getOrdinal(IREE::VM::RodataOp op) const;

  // Returns the byte offset ordinal for a primitive global.
  // Returns -1 if the global is a ref type.
  int64_t getGlobalOrdinal(IREE::Util::GlobalOpInterface op) const;

  // Generic ordinal lookup for any operation with an ordinal.
  int64_t getOrdinal(Operation *op) const;

private:
  OrdinalCounts counts_;
  llvm::DenseMap<Operation *, int64_t> ordinals_;
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_ANALYSIS_ORDINALANALYSIS_H_
