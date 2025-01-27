// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements utilities for lowering StableHLO dialect to Linalg dialect.

#include "compiler/plugins/input/StableHLO/Conversion/LegalizeToLinalgUtils.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

SmallVector<utils::IteratorType, 3>
getParallelAndReductionIterators(unsigned nLoops, unsigned nReduction) {
  SmallVector<utils::IteratorType, 3> res(nLoops - nReduction,
                                          utils::IteratorType::parallel);
  res.append(nReduction, utils::IteratorType::reduction);
  return res;
}

SmallVector<utils::IteratorType, 3>
getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

SmallVector<int64_t> extract1DVector(DenseIntElementsAttr elements) {
  SmallVector<int64_t> ret;
  for (const APInt &element : elements) {
    ret.push_back(element.getLimitedValue());
  }
  return ret;
}

} // namespace mlir::iree_compiler::stablehlo
