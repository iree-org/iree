// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_TILEINFERENCEUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_TILEINFERENCEUTILS_H_

#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler {

/// Walks the producer and consumer chains of the `tilingOp`, and looks for ops
/// that require specific workgroup tile size multiples. Right now, the only ops
/// that require a specific multiple are pack and unpack, since the workgroup
/// tile sizes need to be multiples of the inner_tiles. After walking the IR and
/// finding multiples for the slices of the `tilingOp` operands and results, the
/// function computes and returns the multiples of the `tilingOp` iteration
/// space. The function may fail to find a valid set of workgroup size
/// multiples, in which case the function will fallback to returning a list of
/// all 1, meaning no constraints on the workgroup tile sizes.
SmallVector<int64_t> getWorkgroupSizeMultiples(TilingInterface tilingOp);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_TILEINFERENCEUTILS_H_
