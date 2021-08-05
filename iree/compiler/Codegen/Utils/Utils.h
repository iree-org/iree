// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_UTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_UTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

static constexpr unsigned kNumMaxParallelDims = 3;

/// Returns true if the given `func` is a kernel dispatch entry point.
bool isEntryPoint(FuncOp func);

/// Returns a map from function symbol name to corresponding entry point op.
llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> getAllEntryPoints(
    ModuleOp module);

/// Returns the entry point op for the `funcOp`. Returns `nullptr` on failure.
IREE::HAL::ExecutableEntryPointOp getEntryPoint(FuncOp funcOp);

/// Sets the translation info on the `hal.executable.entry_point` op
/// corresponding to the `entryPointFn`. Returns failure if a translation info
/// is already set on the entry point op and is incompatible with what is being
/// set.
void setTranslationInfo(FuncOp entryPointFn,
                        IREE::HAL::DispatchLoweringPassPipeline passPipeline,
                        ArrayRef<int64_t> workgroupSize = {});

// /// Usually the tile sizes for the first level of tiling decides the
// workgroup
// /// size for the dispatch on the CPU backend. This is a general helper that
// /// converts tile sizes of the first level into workgroup sizes.
// SmallVector<int64_t, 3> getWorkloadPerWorkgroup(
//     ArrayRef<int64_t> firstLevelTileSizes, ArrayRef<int64_t>
//     partitionedLoops);

/// Sets translation for the entry-point function based on op configuration.
LogicalResult setOpConfigAndEntryPointFnTranslation(
    FuncOp entryPointFn, Operation *op, TileSizesListTypeRef tileSizes,
    ArrayRef<int64_t> nativeVectorSizes,
    IREE::HAL::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize = {});

/// Returns the number of outer parallel loops of a linalgOp.
/// Note: To be used only if needed. Use the `getPartitionedLoops` method if
/// this is used to "guess" which loops are distributed.
unsigned getNumOuterParallelLoops(linalg::LinalgOp op);

/// Returns the untiled type of a tiled view for both tensor and memref
/// types. Either walks the `ViewOpInterface` chain (for memrefs) or the
/// `subtensor` op chain (for tensors).
Type getUntiledType(Value tiledView);

/// Returns the untiled type of a tiled view for both tensor and memref
/// types. Either walks the `ViewOpInterface` chain (for memrefs) or the
/// `subtensor` op chain (for tensors).
ArrayRef<int64_t> getUntiledShape(Value tiledView);

/// Assuming that `funcOp` contains a single nested scf.for that represented the
/// tiled+fused+distributed loops with the distribution being across workgroups,
/// i.e.
///
/// scf.for ... {
///   ...
///   scf.for ... {
///     ...
///     filtered_op.
///     ...
///     filtered_op.
///     ...
///   }
/// }
///
/// Returns the list of filtered operations in the functions. If there are no
/// `scf.for` operations in the function return the linalg operations in the
/// body of the function if it has a single basic block. Return failure in all
/// other cases.
using RootOpFilteringFn = std::function<bool(Operation *)>;
LogicalResult getFilteredOps(FuncOp funcOp, RootOpFilteringFn filteringFn,
                             SmallVectorImpl<Operation *> &filteredOps,
                             SmallVectorImpl<Operation *> &tiledLoops);

/// Specialization of `getFilteredOps` for filtering `LinalgOp`s and
/// `LinagExtOp`s.
/// TODO(ravishankarm) This methods also adds the "workgroup" marker to all ops
/// within the loop. The marker is the way to tie into rest of the
/// codegen. Refactor the downstream passes and get rid of the markers once and
/// for all.
LogicalResult getComputeOps(FuncOp funcOp,
                            SmallVectorImpl<Operation *> &computeOps,
                            SmallVectorImpl<Operation *> &tiledLoops);

/// ***Legacy method to be deprecated***
/// Specialization of `getFilteredOps` for filtering `LinalgOp`s
/// TODO(ravishankarm) This methods also adds the "workgroup" marker to all ops
/// within the loop. The marker is the way to tie into rest of the
/// codegen. Refactor the downstream passes and get rid of the markers once and
/// for all.
LogicalResult getLinalgOps(FuncOp funcOp,
                           SmallVectorImpl<linalg::LinalgOp> &computeOps,
                           SmallVectorImpl<Operation *> &tiledLoops);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_UTILS_UTILS_H_
