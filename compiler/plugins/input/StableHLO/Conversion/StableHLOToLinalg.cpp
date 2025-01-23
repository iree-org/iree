// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO/CHLO dialects to Linalg dialect.

#include <memory>

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/TypeConversion.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::stablehlo {

std::unique_ptr<TypeConverter> createStableHloToLinalgTypeConverter() {
  return std::make_unique<LinalgTypeConverter>();
}

} // namespace mlir::iree_compiler::stablehlo
