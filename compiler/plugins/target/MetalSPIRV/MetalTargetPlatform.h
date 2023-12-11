// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_METALTARGETPLATFORM_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_METALTARGETPLATFORM_H_

#include <functional>

namespace mlir::iree_compiler::IREE::HAL {

/// Metal target platforms.
enum class MetalTargetPlatform { macOS, iOS, iOSSimulator };

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_METALTARGETPLATFORM_H_
