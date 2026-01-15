// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/OrdinalAnalysis.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::iree_compiler::IREE::VM {

// Returns the size in bytes of the global when stored in memory.
// Valid only for globals using primitive storage.
static size_t getGlobalStorageSize(IREE::Util::GlobalOpInterface globalOp) {
  auto storageType = globalOp.getGlobalType();
  assert(storageType.isIntOrFloat());
  assert(storageType.getIntOrFloatBitWidth() % 8 == 0);
  return IREE::Util::getRoundedElementByteWidth(storageType);
}

OrdinalAnalysis::OrdinalAnalysis(IREE::VM::ModuleOp moduleOp) {
  // Assign ordinals based on IR order (which should be deterministic).
  int nextFuncOrdinal = 0;
  int nextImportOrdinal = 0;
  int nextExportOrdinal = 0;
  int nextGlobalRefOrdinal = 0;
  int nextRodataOrdinal = 0;

  // Bucket the primitive global ops by byte size for alignment packing.
  SmallVector<SmallVector<IREE::Util::GlobalOpInterface>, 8> primitiveGlobalOps(
      sizeof(int64_t) + 1);

  for (auto &op : moduleOp.getBlock().getOperations()) {
    if (auto funcOp = dyn_cast<IREE::VM::FuncOp>(op)) {
      ordinals_[&op] = nextFuncOrdinal++;
    } else if (isa<IREE::VM::ExportOp>(op)) {
      ordinals_[&op] = nextExportOrdinal++;
    } else if (isa<IREE::VM::ImportOp>(op)) {
      ordinals_[&op] = nextImportOrdinal++;
    } else if (isa<IREE::VM::RodataOp>(op)) {
      ordinals_[&op] = nextRodataOrdinal++;
    } else if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(op)) {
      if (isa<IREE::VM::RefType>(globalOp.getGlobalType())) {
        ordinals_[&op] = nextGlobalRefOrdinal++;
      } else {
        // Bucket the primitive global ops by byte size for alignment packing.
        size_t storageSize = getGlobalStorageSize(globalOp);
        primitiveGlobalOps[storageSize].push_back(globalOp);
      }
    }
  }

  // Assign byte offset values to primitive globals, ensuring that we meet
  // natural alignment requirements on each size type.
  int nextGlobalBytesOrdinal = 0;
  int globalBytes = 0;
  for (auto sizeGlobalOps : llvm::enumerate(primitiveGlobalOps)) {
    size_t storageSize = sizeGlobalOps.index();
    if (sizeGlobalOps.value().empty()) {
      continue;
    }
    nextGlobalBytesOrdinal = llvm::alignTo(nextGlobalBytesOrdinal, storageSize);
    for (auto &globalOp : sizeGlobalOps.value()) {
      ordinals_[globalOp] = nextGlobalBytesOrdinal;
      nextGlobalBytesOrdinal += storageSize;
      globalBytes = std::max(globalBytes, nextGlobalBytesOrdinal);
    }
  }

  // Record counts.
  counts_.importFuncs = nextImportOrdinal;
  counts_.exportFuncs = nextExportOrdinal;
  counts_.internalFuncs = nextFuncOrdinal;
  counts_.globalBytes = globalBytes;
  counts_.globalRefs = nextGlobalRefOrdinal;
  counts_.rodatas = nextRodataOrdinal;
  counts_.rwdatas = 0;
}

int64_t OrdinalAnalysis::getOrdinal(IREE::VM::FuncOp op) const {
  return getOrdinal(op.getOperation());
}

int64_t OrdinalAnalysis::getOrdinal(IREE::VM::ExportOp op) const {
  return getOrdinal(op.getOperation());
}

int64_t OrdinalAnalysis::getOrdinal(IREE::VM::ImportOp op) const {
  return getOrdinal(op.getOperation());
}

int64_t OrdinalAnalysis::getOrdinal(IREE::VM::RodataOp op) const {
  return getOrdinal(op.getOperation());
}

int64_t
OrdinalAnalysis::getGlobalOrdinal(IREE::Util::GlobalOpInterface op) const {
  return getOrdinal(op.getOperation());
}

int64_t OrdinalAnalysis::getOrdinal(Operation *op) const {
  auto it = ordinals_.find(op);
  assert(it != ordinals_.end() && "ordinal not found for operation");
  return it->second;
}

} // namespace mlir::iree_compiler::IREE::VM
