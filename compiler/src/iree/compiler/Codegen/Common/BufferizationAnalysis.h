// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- BufferizationAnalysis.h - Pre bufferization analysis ---------------===//
//
// Analysis to group together tensors within a dispatch region into an
// equivalance class such that all members of a set can be mapped to the same
// memory region.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CODEGEN_COMMON_BUFFERIZATIONANALYSIS_H_
#define IREE_COMPILER_CODEGEN_COMMON_BUFFERIZATIONANALYSIS_H_

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

/// Class that tracks the equivalence relationship between tensors. Its a
/// light-weight wrapper around `llvm::EquivalenceClasses` to account for
/// `Value` not directly supported as a value type by this class.
class BufferizationPlan {
public:
  llvm::EquivalenceClasses<void *>::iterator findValue(Value v) const {
    return mappedTensors.findValue(getPointer(v));
  }

  llvm::EquivalenceClasses<void *>::iterator end() const {
    return mappedTensors.end();
  }

  SmallVector<Value> getTensorsMappedToSameSet(Value v) const {
    SmallVector<Value> tensors;
    for (auto it = mappedTensors.findLeader(getPointer(v)),
              ie = mappedTensors.member_end();
         it != ie; ++it) {
      tensors.push_back(getValue(*it));
    }
    return tensors;
  }

  bool isEquivalent(Value v1, Value v2) const {
    return mappedTensors.isEquivalent(getPointer(v1), getPointer(v2));
  }

  void insert(Value v) { mappedTensors.insert(getPointer(v)); }

  /// Union the sets containing `v1` and `v2`. Checks if the union can be
  /// done and does nothing if union is invalid.
  void unionSets(Value v1, Value v2);

  /// Sets the equivalance class that contains `v` as the set that contains the
  /// result tensor of the dispatch region (i.e. a tensor that is the `value`
  /// operand of a flow.dispatch.tensor.store` op). All operations in this
  /// equivalence class can use the result buffer of the dispatch region to
  /// compute their values in place.
  void storeSet(Value v) { storeLeaders.insert(getLeaderValue(v)); }

  /// Queries if the value `v` is in the same equivalence class as the result of
  /// the dispatch region.
  bool isInStoreSet(Value v) {
    Value leader = getLeaderValue(v);
    if (!leader)
      return false;
    return storeLeaders.count(leader);
  }

  void dump();

private:
  Value getLeaderValue(Value v1) const {
    void *ptr = getPointer(v1);
    auto it = mappedTensors.findLeader(ptr);
    if (it == mappedTensors.member_end()) {
      return nullptr;
    }
    return getValue(*it);
  }

  void *getPointer(Value v) const { return v.getAsOpaquePointer(); }

  Value getValue(void *v) const { return Value::getFromOpaquePointer(v); }

  llvm::EquivalenceClasses<void *> mappedTensors;

  /// Leaders of the sets that contain the result tensor of the dispatch
  /// region, i.e. a tensor that is the `value` operand of a
  /// flow.dispatch.tensor.store` op
  llvm::DenseSet<Value> storeLeaders;
};

/// Analysis the `tensor` values in `funcOp` and groups them together into
/// equivalence classes such that each class contains tensors that can be mapped
/// to the same buffer.
LogicalResult createTensorEquivalenceClasses(mlir::FunctionOpInterface funcOp,
                                             BufferizationPlan &plan);

} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_COMMON_BUFFERIZATIONANALYSIS_H
