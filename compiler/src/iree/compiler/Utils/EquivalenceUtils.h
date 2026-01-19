// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_EQUIVALENCEUTILS_H_
#define IREE_COMPILER_UTILS_EQUIVALENCEUTILS_H_

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

namespace mlir {
class Block;
class IRMapping;
} // namespace mlir

namespace mlir::iree_compiler {

// Returns the runtime value of vscale specified by the user. This is not meant
// to be used for codegen and is meant to circumvent the current limitation on
// host-side querying of this value at runtime. This is temporary until #21317
// is resolved.
unsigned getVscaleValue();

// If the given operation is a `vector.vscale`, generates the constant value for
// it, maps the original op in bvm and adds the operation to the vector of
// vscale ops. Otherwise, it's a noop. This is temporary until #21317 is
// resolved.
void mapVscaleOpToConstant(Operation &op, OpBuilder &builder, IRMapping &bvm,
                           SmallVector<Operation *> &vscaleOps);

// Recursively compares two regions for structural equivalence.
//
// Structural equivalence ensures that operations in both regions
// |lhs| and |rhs| have the same attributes and same use-def structure.
bool isStructurallyEquivalentTo(Region &lhs, Region &rhs);

// Recursively compares two operations for structural equivalence.
//
// Structural equivalence ensures that operations in the regions of both the
// |lhs| and |rhs| have the same attributes and same use-def structure.
bool isStructurallyEquivalentTo(Operation &lhs, Operation &rhs);

// Manages a cache of operation metadata used for efficient structural
// equivalence checks.
class OperationEquivalenceCache {
public:
  explicit OperationEquivalenceCache(MLIRContext *context);
  ~OperationEquivalenceCache();

  bool isSymbolAttrName(StringAttr name) const;

  using IRMappingPtr =
      std::unique_ptr<IRMapping, std::function<void(IRMapping *)>>;
  IRMappingPtr acquireMapping();

  struct RegionEntry {
    llvm::SetVector<Block *> blocks;
  };
  RegionEntry &getRegion(Region *region);

  struct BlockEntry {
    unsigned count = 0;
  };
  BlockEntry &getBlock(Block *block);

  struct OperationEntry {
    NamedAttrList attrs;
  };
  OperationEntry &getOp(Operation *op);

private:
  StringAttr functionRefName; // "function_ref"
  StringAttr symbolAttrName;  // SymbolTable::getSymbolAttrName()

  SmallVector<IRMapping *> mappingFreeList;

  DenseMap<Region *, RegionEntry *> regions;
  DenseMap<Block *, BlockEntry *> blocks;
  DenseMap<Operation *, OperationEntry *> ops;
};

// Recursively compares two operations for structural equivalence.
//
// Structural equivalence ensures that operations in the regions of both the
// |lhs| and |rhs| have the same attributes and same use-def structure.
//
// Uses |cache| to memoize operation information to improve repeated queries.
// Callers must not mutate any IR that may be in the cache between queries.
bool isStructurallyEquivalentTo(OperationEquivalenceCache &cache, Region &lhs,
                                Region &rhs, IRMapping &mapping);
bool isStructurallyEquivalentTo(OperationEquivalenceCache &cache,
                                Operation &lhs, Operation &rhs);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_EQUIVALENCEUTILS_H_
