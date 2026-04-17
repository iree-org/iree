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
