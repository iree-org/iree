// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_VECTOR_LAYOUT_ANALYSIS_H
#define IREE_COMPILER_CODEGEN_VECTOR_LAYOUT_ANALYSIS_H

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
namespace iree_compiler {

using VectorLayoutInterface = IREE::VectorExt::VectorLayoutInterface;

class VectorLayoutAnalysis {
public:
  VectorLayoutAnalysis(Operation *root) : root(root) {}

  template <typename T>
  void setAnchor(Value val, T layout) {
    assert(isa<VectorLayoutInterface>(layout) &&
           "expected layout to implement VectorLayoutInterface");
    anchors[val] = cast<VectorLayoutInterface>(layout);
  }

  LogicalResult run();

  template <typename T>
  T getLayout(Value val) {
    VectorLayoutInterface layout = getLayout(val);
    if (!layout) {
      return T();
    }

    assert(isa<T>(layout) &&
           "expected layout to implement VectorLayoutInterface");
    return cast<T>(layout);
  }

private:
  VectorLayoutInterface getLayout(Value val);

  Operation *root;
  DenseMap<Value, VectorLayoutInterface> anchors;
  DataFlowSolver solver;
};

}; // namespace iree_compiler
}; // namespace mlir

#endif // IREE_COMPILER_CODEGEN_VECTOR_LAYOUT_ANALYSIS_H
