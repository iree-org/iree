// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/MicroKernelOps.h"

#include "iree/builtins/ukernel/exported_bits.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/MicroKernelOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Codegen {

/// Returns true if the dimensions of ShapedType are compatible.
static bool isShapedTypeDimCompatible(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic ||
         lhs == rhs;
}

/// Returns true if the dimensions of ShapedType are compatible.
static bool areShapesCompatible(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return llvm::all_of(llvm::zip(lhs, rhs), [](std::tuple<int64_t, int64_t> it) {
    return isShapedTypeDimCompatible(std::get<0>(it), std::get<1>(it));
  });
}

//===---------------------------------------------------------------------===//
// GenericMicroKernelOp
//===---------------------------------------------------------------------===//

std::pair<int64_t, int64_t> GenericMicroKernelOp::getDpsInitsPositionRange() {
  std::pair<unsigned, unsigned> outsPosAndSize = getODSOperandIndexAndLength(1);
  return {static_cast<int64_t>(outsPosAndSize.first),
          static_cast<int64_t>(outsPosAndSize.first + outsPosAndSize.second)};
}

//===---------------------------------------------------------------------===//
// MMT4DMicroKernelOp
//===---------------------------------------------------------------------===//

std::pair<int64_t, int64_t> Mmt4DMicroKernelOp::getDpsInitsPositionRange() {
  std::pair<unsigned, unsigned> outsPosAndSize = getODSOperandIndexAndLength(2);
  return {static_cast<int64_t>(outsPosAndSize.first),
          static_cast<int64_t>(outsPosAndSize.first + outsPosAndSize.second)};
}

}  // namespace Codegen
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
