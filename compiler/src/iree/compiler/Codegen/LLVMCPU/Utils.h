// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_UTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_UTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

namespace mlir {
namespace iree_compiler {

/// Returns the CPU target features associated with the `targetAttr`, if set.
Optional<StringRef> getCpuFeatures(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Methods to get target information.
bool isX86(IREE::HAL::ExecutableTargetAttr targetAttr);
bool isX86_64(IREE::HAL::ExecutableTargetAttr targetAttr);
bool isAArch64(IREE::HAL::ExecutableTargetAttr targetAttr);
bool isRISCV(IREE::HAL::ExecutableTargetAttr targetAttr);
bool preferIntrinsicsOverAsm(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns true if `targetAttr` has `feature` in its CPU features.
bool hasFeature(IREE::HAL::ExecutableTargetAttr targetAttr, StringRef feature);

/// Returns true if the 'targetAttr' contains '+avx2' in its cpu features.
bool hasAVX2Feature(IREE::HAL::ExecutableTargetAttr targetAttr);

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

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMCPU_UTILS_H_
