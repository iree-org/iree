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

/// The VectorLayoutAnalysis framework is a fixed point iteration analysis
/// that given some anchor points, tries to infer the layout of all values
/// in the program.
///
/// An anchor point is a MLIR Value fixed to a specific layout. Anchor points
/// must be provided by the user. The analysis itself does not have any
/// pre-existing anchor points to start with.
///
/// The analysis does not assume a specific layout, but instead the layout is
/// any Attribute that implements the VectorLayoutInterface.
///
/// The analysis provides a set of rules on operations, that are used to
/// infer the layout of the results and operands of an operation. These rules
/// are defined in the implementation of the analysis.
///
/// To start, the user must define some anchor points using `setAnchor`,
/// anchoring a layout to a specific value. The analysis can then be ran using
/// `run`, at which point it will use it's rules to infer other values in the
/// program. Example:
///
///    %root = vector.transfer_read
///      |
///      --> anchored to layout L
///    %root2 = vector.transfer_read
///    %c = arith.mulf %root, %b
///          |
///          --> %root, %b and %c must have the same layout
///    %e = arith.divf %b, %root2
///          |
///          --> %root2, %b and %e must have the same layout
///
/// Here, the user provided an anchor point for %root, fixing it's layout to L.
/// The layout then uses it's inference rules to find the layout of other
/// values:
///
///    %root = vector.transfer_read
///     |
///     --> infered to layout L
///    %root2 = vector.transfer_read
///     |
///     --> infered to layout L
///    %c = arith.mulf %root, %b
///     |
///     --> infered to layout L
///    %e = arith.divf %b, %root2
///     |
///     --> infered to layout L
///
/// If at any point, a value has a layout, but the user of that value requires
/// a different layout, the analysis inserts a resolution operation. This
/// resolution operation is `iree_vector_ext.layout_conflict_resolution`.
/// For Example:
///
/// %0 = vector.transfer_read
///  |
///  --> anchored to layout L
/// %1 = vector.transfer_read
///  |
///  --> anchored to layout L'
///  arith.addf %0, %1
///     |
///     --> %0 and %1 must have the same layout
///
/// To resolve the conflict, the analysis chooses one of the layouts, say
/// L, and inserts a resolution operation to convert the other layout to L.
///
/// %0 = vector.transfer_read
///  |
///  --> anchored to layout L
/// %1 = vector.transfer_read
///  |
///  --> anchored to layout L'
/// %resolved = iree_vector_ext.layout_conflict_resolution %1
///  |
///  --> infered to layout L
/// arith.addf %0, %resolved
///
/// The analysis itself will not try to resolve the conflict, but instead
/// leaves it to the user to resolve the conflict.
class VectorLayoutAnalysis {
public:
  VectorLayoutAnalysis(Operation *root) : root(root) {}

  /// Fix the layout for a specific value. The layout must implement
  /// VectorLayoutInterface.
  template <typename T>
  void setAnchor(Value val, T layout) {
    assert(isa<VectorLayoutInterface>(layout) &&
           "expected layout to implement VectorLayoutInterface");
    auto typedVal = dyn_cast<TypedValue<VectorType>>(val);
    assert(typedVal && "expected value to be a vector type");
    anchors[typedVal] = cast<VectorLayoutInterface>(layout);
  }

  /// Run the analysis. The analysis expects that the user has set some anchor
  /// points and is trying to infer the layout of other values.
  LogicalResult run();

  /// Get the infered layout of a specific value. This should only be called
  /// after the analysis has been run.
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

  /// Return the operation this on which this analysis was rooted on.
  Operation *getRootOperation() const;

  /// Annotate each operation with "vector_result_x" attributes that specify
  /// the layout of each result of the operation. 'x' here is the x^th result.
  /// This is only for debugging purposes, to understand the result of the
  /// analysis better.
  void debugAnnotateLayouts();

  void print(raw_ostream &os);
  void dump();

private:
  VectorLayoutInterface getLayout(Value val);

  Operation *root;
  DenseMap<TypedValue<VectorType>, VectorLayoutInterface> anchors;
  DataFlowSolver solver;
};

void setAnchorOpsFromAttributes(VectorLayoutAnalysis &analysis,
                                Operation *root);

}; // namespace iree_compiler
}; // namespace mlir

#endif // IREE_COMPILER_CODEGEN_VECTOR_LAYOUT_ANALYSIS_H
