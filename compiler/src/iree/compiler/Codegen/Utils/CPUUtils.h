// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_

#include "iree/compiler/Codegen/Utils/Utils.h"

namespace mlir::iree_compiler {

/// Find the root operation for the dispatch region given `computeOps` that are
/// obtained by a post order walk, i.e. in presence of nested compute ops the
/// outermost operations are towards the end of the list. The priority is:
///   1. A Linalg operation that has reduction loops.
///   2. Any other Linalg op or LinalgExt op.
///   3. An operation that implements TilingInterface.
/// If there are multiple operations meeting the same priority, the one closer
/// to the end of the function is the root op.
FailureOr<Operation *> getRootOperation(ArrayRef<Operation *> computeOps);

/// Creates a string attribute containing the name of the attribute that is
/// used to enable decomposition.
StringAttr getEnableDecompositionAttrName(MLIRContext *ctx);
std::string getEnableDecompositionStr();

/// Creates a string attribute containing the name of the attribute that is
/// used to enable loop peeling.
StringAttr getEnableLoopPeelingAttrName(MLIRContext *ctx);
std::string getEnableLoopPeelingStr();

/// Returns true if the UnitAttr of the `label` is enabled for the input
/// function. This is is inferred from the config dictionary. attribute that's
/// part of to the translation info corresponding to this function.
bool isOptEnabled(FunctionOpInterface funcOp, StringRef label);

/// Returns if scalable vectorization is enabled or not.
bool isScalableVectorizationEnabled();

/// Returns whether Armv9-A streaming SVE mode is forced for dispatch regions
/// containing scalable vector operations, via
/// `--iree-llvmcpu-force-arm-streaming`.
bool isArmStreamingForced();

/// Returns the runtime value of vscale specified by the user. This is not meant
/// to be used for codegen and is meant to circumvent the current limitation on
/// host-side querying of this value at runtime. This is temporary until #21317
/// is resolved.
unsigned getUserVscaleValue();

/// Returns true if `op` is a direct producer of `rootOp`, i.e., at least one
/// of `op`'s results is used as an operand of `rootOp`.
bool isProducerOfRootOp(Operation *op, Operation *rootOp);

/// Returns true if the 'targetAttr' contains '+sve' or '+sve2' in its cpu
/// features or any other feature flag that includes them.
bool hasAnySVEFeature(DictionaryAttr targetConfig);

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

/// Returns true if target supports scalable vector code generation.
bool targetSupportsScalableVectors(DictionaryAttr targetConfig);

/// Reads the vscale range from the target config, if present.
std::optional<std::pair<int64_t, int64_t>>
getConfigVscaleRange(DictionaryAttr targetConfig);

/// Records the `[vscaleMin, vscaleMax]` scalable-vector range on the `config`
/// list, in vscale units.
void addConfigVscaleRange(MLIRContext *context, int64_t vscaleMin,
                          int64_t vscaleMax,
                          SmallVectorImpl<NamedAttribute> &config);

/// Returns the default vscale range for the given target, independent of any
/// user-specified range. Currently only returns a range for AArch64 targets
/// with SVE/SVE2 enabled.
std::optional<vector::VscaleRange>
getDefaultVscaleRange(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns the effective scalable-vector range for the given target: a
/// user-specified range recorded on the target config if present, otherwise the
/// target's default range, or nullopt if neither exists.
std::optional<vector::VscaleRange>
getVscaleRange(IREE::HAL::ExecutableTargetAttr targetAttr);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_
