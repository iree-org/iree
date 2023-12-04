// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREECodegenAttrs.h - Codegen dialect attributes --------------------===//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_
#define IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {
/// Typedef for tile sizes to use at different levels of tiling.
using TileSizesListType = SmallVector<SmallVector<int64_t>>;
using TileSizesListTypeRef = ArrayRef<SmallVector<int64_t>>;
/// Typedef for scalable tile flags at different levels of tiling.
using ScalableTileFlagsListType = SmallVector<SmallVector<bool>>;
using ScalableTileFlagsListTypeRef = ArrayRef<SmallVector<bool>>;
} // namespace iree_compiler
} // namespace mlir

// clang-format off
#include "iree/compiler/Codegen/Dialect/LoweringConfigEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h.inc"
// clang-format on

namespace mlir {
namespace iree_compiler {
//===----------------------------------------------------------------------===//
// Helpers for getting/setting iree_codegen.translation_info attribute on the
// `hal.executable.export`
//===----------------------------------------------------------------------===//

/// Gets the translate executable info attribute value associated with
/// `exportOp`. It expects that the attribute is stored using the identifier
/// `translation_info`.
IREE::Codegen::TranslationInfoAttr
getTranslationInfo(IREE::HAL::ExecutableExportOp exportOp);
/// Returns the translation info for the `funcOp` (by looking at the entry
/// point). Returns `nullptr` on failure.
inline IREE::Codegen::TranslationInfoAttr
getTranslationInfo(func::FuncOp funcOp) {
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (failed(exportOp))
    return nullptr;
  return getTranslationInfo(*exportOp);
}

/// Returns the identical TranslationInfoAttr. Returns nullptr if entry point
/// functions have different TranslationInfoAttr.
/// There might be multiple entry points in the module. Currently, all of them
/// need to have the same translation info.
/// TODO(ravishankarm): This is strange that this is not enforced
/// structurally, but something to address later on. The main issue is how
/// to invoke separate dynamic pass pipelines on  entry point functions,
/// when the passes might have module level changes. For now this
/// restriction is fine.
std::optional<IREE::Codegen::TranslationInfoAttr>
getIdenticalTranslationInfo(IREE::HAL::ExecutableVariantOp variantOp);

// TODO(ravishankarm, benvanik): Eventually all the information needed for the
// lowering will be consolidated into a single attribute with richer
// information.

/// Returns the workgroup size specified on the `exportOp`.
SmallVector<int64_t> getWorkgroupSize(IREE::HAL::ExecutableExportOp exportOp);

/// Returns the subgroup size specified on the `exportOp`.
std::optional<int64_t> getSubgroupSize(IREE::HAL::ExecutableExportOp exportOp);

/// Sets and overwrites the dispatch workgroup/subgroup size for the given entry
/// point function. Returns failure if the given entry point is not exported via
/// hal.executable.export.
LogicalResult setDispatchConfig(func::FuncOp entryPoint,
                                ArrayRef<int64_t> workgroupSize,
                                std::optional<int64_t> subgroupSize);

/// Sets and overwites the translate executable info for the given entry point.
/// Returns failure if the given entry point is not exported via
/// hal.executable.export.
LogicalResult
setTranslationInfo(func::FuncOp entryPoint,
                   IREE::Codegen::TranslationInfoAttr translationInfo);

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.lowering_config` attribute on root
// operations.
//===----------------------------------------------------------------------===//

/// Returns the op that carries the `lowering_config` attribute; returns nullptr
/// if none is carrying the attribute.
///
/// This scans ops in top-down order and the first one carrying the attribute
/// will be returned.
FailureOr<Operation *>
getLoweringConfigCarryingOp(ArrayRef<Operation *> computeOps);

/// Returns the lowering configuration set for an operation. Returns `nullptr`
/// if no value is set.  It expects that the attribute is stored using the
/// identifier `lowering_config`.
IREE::Codegen::LoweringConfigAttr getLoweringConfig(Operation *op);

/// Returns the lowering configuration from the list of operations; returns
/// nullptr if unable to find.
///
/// This scans ops in top-down order and the first one carrying the attribute
/// will be returned.
FailureOr<IREE::Codegen::LoweringConfigAttr>
getLoweringConfig(ArrayRef<Operation *> computeOps);

/// Returns the tile sizes for a particular operation if the
/// `iree_codegen.lowering_config` attribute is set on it.
SmallVector<int64_t> getTileSizes(Operation *op, unsigned level);
SmallVector<Value> getTileSizes(OpBuilder &b, Operation *op, unsigned level);

/// Returns the number of tiling levels defined in the
/// `iree_codegen.lowering_config` of this operation.
unsigned getNumTileLevels(Operation *op);

/// Sets the lowering configuration, overwriting existing attribute values.
void setLoweringConfig(Operation *op, IREE::Codegen::LoweringConfigAttr config);

/// Convenience function that sets the lowering configuration on the operation
/// and translation info on the entry point op for the common case of specifying
/// tile sizes to use for the operation, and pass pipeline to use for the
/// translation.
inline LogicalResult setOpConfigAndEntryPointFnTranslation(
    func::FuncOp entryPointFn, Operation *op, TileSizesListTypeRef tileSizes,
    ScalableTileFlagsListTypeRef scalableTileFlags,
    IREE::Codegen::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize = {},
    std::optional<int64_t> subgroupSize = {},
    unsigned softwarePipelineDepth = 0,
    unsigned softwarePipelineStoreStage = 1) {
  MLIRContext *context = entryPointFn.getContext();
  auto config = IREE::Codegen::LoweringConfigAttr::get(context, tileSizes,
                                                       scalableTileFlags);
  setLoweringConfig(op, config);
  if (failed(setDispatchConfig(entryPointFn, workgroupSize, subgroupSize)))
    return failure();
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn.getContext(), passPipeline, softwarePipelineDepth,
      softwarePipelineStoreStage);
  return setTranslationInfo(entryPointFn, translationInfo);
}

/// Overload of setOpConfigAndEntryPointFnTranslation() for the "no scalable
/// flags" case.
inline LogicalResult setOpConfigAndEntryPointFnTranslation(
    func::FuncOp entryPointFn, Operation *op, TileSizesListTypeRef tileSizes,
    IREE::Codegen::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize = {},
    std::optional<int64_t> subgroupSize = {},
    unsigned softwarePipelineDepth = 0,
    unsigned softwarePipelineStoreStage = 1) {
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes, {}, passPipeline, workgroupSize,
      subgroupSize, softwarePipelineDepth, softwarePipelineStoreStage);
}

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.compilation_info` attribute on root
// operations to override IREEs default compilation.
//===----------------------------------------------------------------------===//

/// Returns the `#iree_codegen.compilation_info` set on the operation. Assumes
/// that the identifier used is `compilation_info`.
IREE::Codegen::CompilationInfoAttr getCompilationInfo(Operation *op);

/// Sets the `config` to use for compiling the operation. If `op` is the root
/// operation of the dispatch region, overrides the default configuration that
/// is used for compilation.
void setCompilationInfo(Operation *op,
                        IREE::Codegen::CompilationInfoAttr config);

/// Removes the `#iree_codegen.compilation_info` attribute that is set on the
/// operation.
void eraseCompilationInfo(Operation *op);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_
