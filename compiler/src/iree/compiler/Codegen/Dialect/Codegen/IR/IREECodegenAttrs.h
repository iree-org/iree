// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREECodegenAttrs.h - Codegen dialect attributes --------------------===//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_
#define IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

/// Typedef for tile sizes to use at different levels of tiling.
using TileSizesListType = SmallVector<SmallVector<int64_t>>;
using TileSizesListTypeRef = ArrayRef<SmallVector<int64_t>>;
/// Typedef for scalable tile flags at different levels of tiling.
using ScalableTileFlagsListType = SmallVector<SmallVector<bool>>;
using ScalableTileFlagsListTypeRef = ArrayRef<SmallVector<bool>>;
/// Returns whether tuner attributes should be set on root ops.
bool shouldSetTunerAttributes();
/// Returns whether pipeline constraints should be emitted for root ops.
bool shouldEmitPipelineConstraints();
} // namespace mlir::iree_compiler

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h.inc"
// clang-format on

namespace mlir::iree_compiler::IREE::Codegen {

/// Callback type for VMVX pipeline builders.
using VMVXPipelineBuilder =
    LogicalResult (*)(Attribute pipelineAttr, OpPassManager &pm,
                      const CodegenPipelineOptions *options);

/// Registers the VMVX pipeline builder callback. Must be called before
/// any compilation that uses #iree_codegen.vmvx_pipeline.
void registerVMVXPipelineBuilder(VMVXPipelineBuilder builder);

} // namespace mlir::iree_compiler::IREE::Codegen

namespace mlir::iree_compiler {
//===----------------------------------------------------------------------===//
// Constant names.
//===----------------------------------------------------------------------===//
constexpr StringLiteral kConfigAttrName = "lowering_config";
constexpr StringLiteral kTuningSpecDefaultEntrypointAttrName =
    "iree_codegen.tuning_spec_with_default_entrypoint";
constexpr StringLiteral kTuningSpecEntrypointAttrName =
    "iree_codegen.tuning_spec_entrypoint";
constexpr StringLiteral kSerializedTuningSpecAttrName =
    "iree_codegen.tuning_spec_mlirbc";
constexpr StringLiteral kKernelConfigSpecName = "__kernel_config";
constexpr StringLiteral kUkernelAttrName = "iree_codegen.ukernel";
constexpr StringLiteral kUKernelProviderName = "iree_codegen.ukernel_provider";
constexpr StringLiteral kVectorTileSizesAttrName =
    "iree_codegen.vector_tile_sizes";

//===----------------------------------------------------------------------===//
// Helpers for getting/setting iree_codegen.translation_info attribute on a
// FunctionOpInterface op.
//===----------------------------------------------------------------------===//

/// Returns the translation info for the `funcOp`. Returns `nullptr` on failure.
IREE::Codegen::TranslationInfoAttr
getTranslationInfo(mlir::FunctionOpInterface funcOp);

/// Returns the workgroup size specified on the `exportOp`.
std::optional<SmallVector<int64_t>>
getWorkgroupSize(mlir::FunctionOpInterface funcOp);

/// Returns the subgroup size specified on the `exportOp`.
std::optional<int64_t> getSubgroupSize(mlir::FunctionOpInterface funcOp);

/// Sets and overwrites the translate executable info for the given entry point.
/// Returns success() at the end. It is convenient when a caller need to
/// propagate the state.
LogicalResult
setTranslationInfo(mlir::FunctionOpInterface entryPoint,
                   IREE::Codegen::TranslationInfoAttr translationInfo);

/// Erases any translation info set on an operation.
void eraseTranslationInfo(mlir::FunctionOpInterface funcOp);

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.lowering_config` attribute on root
// operations.
//===----------------------------------------------------------------------===//

/// Returns the lowering configuration set for an operation. Returns `nullptr`
/// if no value is set.  It expects that the attribute is stored using the
/// identifier `lowering_config`.
template <typename ConfigTy = IREE::Codegen::LoweringConfigAttrInterface>
ConfigTy getLoweringConfig(Operation *op) {
  return op->getAttrOfType<ConfigTy>(kConfigAttrName);
}

/// Returns the op that carries the `lowering_config` attribute; returns nullptr
/// if none is carrying the attribute.
///
/// This scans ops in top-down order and the first one carrying the attribute
/// will be returned.
template <typename ConfigTy = IREE::Codegen::LoweringConfigAttrInterface>
FailureOr<Operation *>
getLoweringConfigCarryingOp(ArrayRef<Operation *> computeOps) {
  for (Operation *op : computeOps) {
    if (getLoweringConfig<ConfigTy>(op)) {
      return op;
    }
  }
  return failure();
}

/// Returns the lowering configuration from the list of operations; returns
/// failure if no operations is carrying a lowering config.
///
/// This scans ops in top-down order and the first one carrying the attribute
/// will be returned.
template <typename ConfigTy = IREE::Codegen::LoweringConfigAttrInterface>
FailureOr<ConfigTy> getFirstLoweringConfig(ArrayRef<Operation *> computeOps) {
  FailureOr<Operation *> op = getLoweringConfigCarryingOp<ConfigTy>(computeOps);
  if (failed(op)) {
    return failure();
  }
  return getLoweringConfig<ConfigTy>(*op);
}

/// Returns the tile sizes for a particular operation if the
/// `iree_codegen.lowering_config` attribute is set on it.
SmallVector<int64_t> getTileSizes(Operation *op, unsigned level);
SmallVector<Value> getTileSizes(OpBuilder &b, Operation *op, unsigned level);

/// Sets the lowering configuration, overwriting existing attribute values.
void setLoweringConfig(Operation *op, Attribute config);

/// Sets an attribute to identify the root op. The `set` parameter groups root
/// ops into numbered sets (default 0): all root ops in the same set share the
/// same `lowering_config`. Codegen does not rely on this attribute; it is only
/// used for constraint generation when tuning.
void setRootOpInfo(Operation *op, int64_t set = 0);

bool hasRootOpInfo(Operation *op);
IREE::Codegen::RootOpAttr getRootOpInfo(Operation *op);

/// Convenience function that sets the lowering configuration on the operation
/// and translation info.
inline LogicalResult setOpConfigAndEntryPointFnTranslation(
    mlir::FunctionOpInterface entryPointFn, Operation *op,
    IREE::Codegen::LoweringConfigAttrInterface config,
    IREE::Codegen::TranslationInfoAttr translationInfo) {
  if (shouldSetTunerAttributes()) {
    setRootOpInfo(op);
  }
  if (config) {
    setLoweringConfig(op, config);
  }
  if (translationInfo) {
    (void)setTranslationInfo(entryPointFn, translationInfo);
  }
  return success();
}

/// Function to erase lowering configs that are set on an operation.
void eraseLoweringConfig(Operation *op);

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

//===----------------------------------------------------------------------===//
// Helpers for getting/setting attributes related to ukernels.
//===----------------------------------------------------------------------===//

/// Returns the `iree_codegen.ukernel_provider` in the provided dictionary if
/// present.
IREE::Codegen::UKernelProviderInterface
getUKernelProviderFromTarget(DictionaryAttr dict);

/// Returns the `iree_codegen.ukernel` on the operation.
IREE::Codegen::UKernelDescriptorAttr getUKernelDescriptor(Operation *op);

/// Sets the `iree_codegen.ukernel` on the operation.
void setUKernelDescriptor(Operation *op,
                          IREE::Codegen::UKernelDescriptorAttr descriptor);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG_H_
