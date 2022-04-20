// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionUtils.h - Utilities that are useful for fusion -------------===//
//
// Declares utility functions and analyses that are useful across passes
// to help with fusion before dispatch region formation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Returns true if the `use` is from a producer linalg op that can be fused
/// with the consumer linalg op using tile + fuse.
bool areLinalgOpsFusableUsingTileAndFuse(OpOperand &use);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
