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

/// Methods to retrieve information association with `configuration` field
/// of `hal.executable.target` attribute used commonly in CPU codegen pipelines.
std::optional<int64_t>
getConfigMaxStackAllocationSize(DictionaryAttr targetConfig);
std::optional<int64_t> getConfigNativeVectorSize(DictionaryAttr targetConfig);

/// Methods to add attributes to the `config` list.
void addConfigMaxStackAllocationSize(MLIRContext *context,
                                     int64_t maxStackAllocationSize,
                                     SmallVectorImpl<NamedAttribute> &config);
void addConfigNativeVectorSize(MLIRContext *context, int64_t nativeVectorSize,
                               SmallVectorImpl<NamedAttribute> &config);

bool preferIntrinsicsOverAsm(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+avx2' in its cpu features.
bool hasAVX2Feature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+avx512f' in its cpu features.
bool hasAVX512fFeature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+v' in its cpu features.
bool hasVFeature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+zve32x' in its cpu features.
bool hasZve32xFeature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+zve32f' in its cpu features.
bool hasZve32fFeature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+zve64x' in its cpu features.
bool hasZve64xFeature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains any riscv vector feature in its
/// cpu features.
bool hasAnyVFeature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+sve' or '+sve2' in its cpu
/// features or any other feature flag that includes them.
bool hasAnySVEFeature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+sme' in its cpu features.
bool hasSMEFeature(DictionaryAttr targetConfig);

/// Returns true if the 'targetAttr' contains '+i8mm' in its cpu features.
bool hasI8mmFeature(DictionaryAttr targetConfig);

/// Returns true if the `genericOp` is a simple 2D transpose, i.e.,
///
///   1. The op has 2 dimensions.
///   2. The op has a single input and a single output.
///   3. One of the `indexing_maps` is a permutation and the other an identity.
///   4. The body of the generic has a single yield op returning the block
///   argument corresponding to the input.
bool isLinalgGeneric2DTranspose(linalg::GenericOp genericOp);

/// Returns true if the op could result in undefined behavior.
bool mayHaveUndefinedBehaviorInMasking(Operation *op);

int64_t
getMaxVectorSizeForLargeVectorCheck(IREE::HAL::ExecutableTargetAttr targetAttr);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_UTILS_H_
