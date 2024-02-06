// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_INDEXING_PASSES
#define IREE_DIALECT_INDEXING_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Debug/test passes
//===----------------------------------------------------------------------===//

def TestIndexRangeAnalysis : Pass<"iree-indexing-test-index-range-analysis", ""> {
  let summary = "Tests integer/index range analysis.";
  let description = [{
    Tests index range analysis by evaluating any
    'iree_unregistered.test_intrange' ops and setting the results on an
    attribute.
  }];
  let constructor = [{
    mlir::iree_compiler::IREE::Indexing::createTestIndexRangeAnalysisPass()
  }];
}

#endif  // IREE_DIALECT_INDEXING_PASSES
