// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler {

/// Contraction-like dimension classification used by both matmul and conv.
struct ContractionLikeDims {
  SmallVector<unsigned> m;
  SmallVector<unsigned> n;
  SmallVector<unsigned> k;
};

/// Problem size, loop count, and indexing maps for a root op.
struct RootOpLoopInfo {
  SmallVector<int64_t> staticLoopRanges;
  unsigned numLoops;
  SmallVector<AffineMap> indexingMaps;
};

// Knobs dict keys.
constexpr StringLiteral kKnobWorkgroupKey = "workgroup";
constexpr StringLiteral kKnobReductionKey = "reduction";
constexpr StringLiteral kKnobMmaKindKey = "mma_kind";
constexpr StringLiteral kKnobSubgroupBasisKey = "subgroup_basis";
constexpr StringLiteral kKnobWorkgroupSizeKey = "workgroup_size";
constexpr StringLiteral kKnobSubgroupSizeKey = "subgroup_size";
// For subgroup basis subdict.
constexpr StringLiteral kKnobCountsKey = "counts";
constexpr StringLiteral kKnobMappingKey = "mapping";

// Knob variable names.
constexpr StringLiteral kKnobMmaIdxName = "mma_idx";
constexpr StringLiteral kKnobSgMCntName = "sg_m_cnt";
constexpr StringLiteral kKnobSgNCntName = "sg_n_cnt";
constexpr StringLiteral kKnobSgSizeName = "sg_size";
constexpr StringLiteral kKnobWgSizeXName = "wg_size_x";
constexpr StringLiteral kKnobWgSizeYName = "wg_size_y";
constexpr StringLiteral kKnobWgSizeZName = "wg_size_z";

// Knob variable name prefixes. The loop dim count varies
// per problem, so names are built at runtime as prefix + dim idx.
constexpr StringLiteral kKnobWgPrefix = "wg_";
constexpr StringLiteral kKnobRedPrefix = "red_";

/// Build a knob variable name from a prefix, e.g. ("wg_", 2) -> "wg_2".
std::string makeVarName(StringRef prefix, unsigned idx);

/// Get unique compatible MMA attrs for matmul and conv ops.
SmallVector<Attribute> getCompatibleMMAAttrs(linalg::LinalgOp op,
                                             IREE::GPU::TargetAttr gpuTarget,
                                             const RootOpLoopInfo &loopInfo,
                                             const ContractionLikeDims &dims);

/// Get contraction-like (m,n,k) dims for a linalg op.
FailureOr<ContractionLikeDims>
inferContractionLikeDims(linalg::LinalgOp linalgOp);

/// Returns loop info for supported root ops.
std::optional<RootOpLoopInfo> getRootOpLoopInfo(Operation *rootOp);

/// Build the VectorDistribute knobs dict for contraction-like dims.
DictionaryAttr
buildVectorDistributeKnobsDict(MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
                               const ContractionLikeDims &dims,
                               ArrayRef<Attribute> compatibleMMAs);

/// LLVMGPU constraint emitter callback. Suitable for use as a
/// GPUConstraintEmitter registered via registerGPUPipelineCallbacks.
LogicalResult emitLLVMGPUConstraints(Attribute pipelineAttr,
                                     ArrayRef<Operation *> rootOps);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_
