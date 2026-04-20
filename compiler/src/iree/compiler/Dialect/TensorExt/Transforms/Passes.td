// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_TENSOREXT_PASSES
#define IREE_DIALECT_TENSOREXT_PASSES

include "mlir/Pass/PassBase.td"

def TestSparseOpInterfaceMethodsPass :
    InterfacePass<"iree-tensor-ext-test-sparse-op-interface-methods",
                  "mlir::FunctionOpInterface"> {
  let summary = "Test SparseCastOpInterface methods pass.";
  let options = [
    Option<"testLowerLoopRange", "test-lower-loop-range", "bool",
           /*default=*/"false", "Test lowerLoopRange method">,
    Option<"testGetEstimatedLoopRange", "test-get-estimated-loop-range", "bool",
           /*default=*/"false", "Test getEstimatedLoopRange method">,
    Option<"testResolveRange", "test-resolve-range", "bool",
           /*default=*/"false", "Test resolveRange method">,
    Option<"testGetDistributionInfo", "test-get-distribution-info", "bool",
           /*default=*/"false",
           "Test getDistributionInfoForSparseDimensions method">,
  ];
}

#endif // IREE_DIALECT_TENSOREXT_PASSES
