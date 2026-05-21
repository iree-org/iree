// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUSMTKNOBRESOLUTION_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUSMTKNOBRESOLUTION_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"

#include <optional>

namespace mlir::iree_compiler::IREE::GPU {

using KnobAssignmentMap = llvm::DenseMap<llvm::StringRef, int64_t>;

/// Walk a knobs template dictionary and extract concrete values from a
/// config dictionary with matching structure.
LogicalResult extractKnobValues(DictionaryAttr knobsTemplate,
                                DictionaryAttr configAttrs,
                                DenseMap<StringAttr, int64_t> &result);

/// Build a flat knobs lookup dict from GPU lowering config and translation
/// info for SMT knob extraction.
DictionaryAttr
buildKnobLookupDictFromGPUConfig(LoweringConfigAttr gpuConfig,
                                 Codegen::TranslationInfoAttr translationInfo);

/// Overlay explicit knob assignments on values from the existing dispatch
/// config. Returns std::nullopt when no matching GPU config is available.
std::optional<KnobAssignmentMap>
mergeKnobAssignmentsWithExistingGPUConfig(Codegen::ConstraintsOp constraintsOp,
                                          const KnobAssignmentMap &assignments);

/// Overlay materialized knobs on the existing dispatch config lookup dict.
/// Returns `materializedKnobs` when no matching GPU config is available.
DictionaryAttr mergeMaterializedKnobsWithExistingDispatchConfig(
    Codegen::ConstraintsOp constraintsOp, DictionaryAttr materializedKnobs);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUSMTKNOBRESOLUTION_H_
