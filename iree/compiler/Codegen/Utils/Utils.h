// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_UTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_UTILS_H_

#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

static constexpr unsigned kNumMaxParallelDims = 3;

//===----------------------------------------------------------------------===//
// Utility functions to get entry points
//===----------------------------------------------------------------------===//

/// Returns true if the given `func` is a kernel dispatch entry point.
bool isEntryPoint(FuncOp func);

/// Returns a map from function symbol name to corresponding entry point op.
llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> getAllEntryPoints(
    ModuleOp module);

/// Returns the entry point op for the `funcOp`. Returns `nullptr` on failure.
IREE::HAL::ExecutableEntryPointOp getEntryPoint(FuncOp funcOp);

inline bool isVMVXBackend(IREE::HAL::ExecutableVariantOp variantOp) {
  return variantOp.target().getBackend().getValue() == "vmvx";
}
inline bool isVMVXBackend(FuncOp entryPointFn) {
  auto variantOp =
      entryPointFn->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  return isVMVXBackend(variantOp);
}

//===----------------------------------------------------------------------===//
// Utility functions to get untiled op shapes
//===----------------------------------------------------------------------===//

/// Returns the untiled type of a tiled view for both tensor and memref
/// types. Either walks the `ViewOpInterface` chain (for memrefs) or the
/// extract/load op chain (for tensors).
ArrayRef<int64_t> getUntiledShape(Value tiledView);

/// Returns the untiled result shape for the given Linalg `op` by inspecting
/// the subview chain or the tiled and distributed loop nests around it.
SmallVector<int64_t> getUntiledResultShape(linalg::LinalgOp linalgOp,
                                           unsigned resultNum);

//===----------------------------------------------------------------------===//
// Utility functions to set configurations
//===----------------------------------------------------------------------===//

/// Returns the loops that are partitioned during dispatch region formations, in
/// order, i.e. starting from the outer-most to innermost.
/// Note that this is the same method that is used at the Flow dispatch region
/// formation to tile and distribute the ops.
// TODO: This method is to be deprecated.
SmallVector<unsigned> getPartitionedLoops(Operation *op);

/// Return the tile sizes to use for the Flow partitioned loops given the
/// workload per workgroup. The tile sizes for the partitioned loops are
/// obtained from the workload per workgroup. The other loops are returned as
/// zero.
SmallVector<int64_t> getDistributedTileSizes(
    IREE::Flow::PartitionableLoopsInterface interfaceOp,
    ArrayRef<int64_t> workloadPerWorkgroup);

/// Information about a tiled and distributed loop.
///
/// Right now distribution is happening as the same time when we tile the linalg
/// op. 0) Given an original loop:
///
/// ```
/// scf.for %iv = %init_lb to %init_ub step %init_step { ... }
/// ```
//
/// 1) After tiling with tile size `%tile_size`, we have:
//
/// ```
/// %tiled_step = %init_step * %tile_size
/// scf.for %iv = %init_lb to %init_ub step %tiled_step { ... }
/// ```
///
/// 2) After distribution with processor `%id` and `%count`, we have:
//
/// ```
/// %dist_lb = %init_lb + %id * %tiled_step
/// %dist_step = %init_step * %tile_size * %count
/// scf.for %iv = %dist_lb to %init_ub step %dist_step { ... }
/// ```
///
/// Given a loop already after 2), this struct contains recovered information
/// about 0) and 1).
struct LoopTilingAndDistributionInfo {
  // The tiled and distributed loop.
  Operation *loop;
  // The lower bound for the original untiled loop.
  OpFoldResult untiledLowerBound;
  // The upper bound for the original untiled loop.
  OpFoldResult untiledUpperBound;
  // The step for the original untiled loop.
  OpFoldResult untiledStep;
  // The tile size used to tile (and not distribute) the original untiled loop.
  Optional<int64_t> tileSize;
  // The processor dimension this loop is distributed to.
  unsigned processorDistributionDim;
};

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
LogicalResult getFilteredOps(
    FuncOp funcOp, RootOpFilteringFn filteringFn,
    SmallVectorImpl<Operation *> &filteredOps,
    SmallVectorImpl<LoopTilingAndDistributionInfo> &tiledLoops);

/// Specialization of `getFilteredOps` for filtering `LinalgOp`s and
/// `LinagExtOp`s.
LogicalResult getComputeOps(
    FuncOp funcOp, SmallVectorImpl<Operation *> &computeOps,
    SmallVectorImpl<LoopTilingAndDistributionInfo> &tiledLoops);

/// If the given `forOp` is a tiled and distributed loop, returns its tiling and
/// distribution information.
Optional<LoopTilingAndDistributionInfo> isTiledAndDistributedLoop(
    scf::ForOp forOp);

/// Collects information about loops matching tiled+distribute pattern.
SmallVector<LoopTilingAndDistributionInfo> getTiledAndDistributedLoopInfo(
    FuncOp funcOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_UTILS_UTILS_H_
