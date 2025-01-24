// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Utils for lowering of the StableHLO dialect to the Linalg dialect.

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_LEGALIZE_TO_LINALG_UTILS_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_LEGALIZE_TO_LINALG_UTILS_H_

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<utils::IteratorType, 3>
getParallelAndReductionIterators(unsigned nLoops, unsigned nReduction);

/// Returns an ArrayAttr that contains `nParallelLoops` "parallel" attributes.
SmallVector<utils::IteratorType, 3>
getNParallelLoopsAttrs(unsigned nParallelLoops);

/// Extracts integer values from the attribute |elements|.
SmallVector<int64_t> extract1DVector(DenseIntElementsAttr elements);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_LEGALIZE_TO_LINALG_UTILS_H_
