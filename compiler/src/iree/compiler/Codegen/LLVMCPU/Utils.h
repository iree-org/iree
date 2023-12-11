// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_UTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_UTILS_H_

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

namespace mlir::iree_compiler {

bool preferIntrinsicsOverAsm(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if the 'targetAttr' contains '+avx2' in its cpu features.
bool hasAVX2Feature(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if the 'targetAttr' contains '+avx512f' in its cpu features.
bool hasAVX512fFeature(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if the 'targetAttr' contains '+v' in its cpu features.
bool hasVFeature(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if the 'targetAttr' contains '+zve32x' in its cpu features.
bool hasZve32xFeature(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if the 'targetAttr' contains '+zve32f' in its cpu features.
bool hasZve32fFeature(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if the 'targetAttr' contains '+zve64x' in its cpu features.
bool hasZve64xFeature(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if the 'targetAttr' contains '+sve' or '+sve2' in its cpu
/// features.
bool hasAnySVEFeature(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if the 'targetAttr' contains '+sme' in its cpu features.
bool hasSMEFeature(IREE::HAL::ExecutableTargetAttr targetAttr);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_UTILS_H_
