// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_UTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_UTILS_H_

#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace iree_compiler {

static constexpr unsigned kNumMaxParallelDims = 3;

//===----------------------------------------------------------------------===//
// Utility functions to get entry points
//===----------------------------------------------------------------------===//

/// Returns true if the given `func` is a kernel dispatch entry point.
bool isEntryPoint(func::FuncOp func);

/// Returns a map from function symbol name to corresponding entry point op.
llvm::StringMap<IREE::HAL::ExecutableExportOp> getAllEntryPoints(
    ModuleOp module);

/// Returns the entry point op for the `funcOp`. Returns `nullptr` on failure.
FailureOr<IREE::HAL::ExecutableExportOp> getEntryPoint(func::FuncOp funcOp);

/// Returns the ExecutableVariableOp enclosing `op`. Returns `nullptr` on
/// failure.
FailureOr<IREE::HAL::ExecutableVariantOp> getExecutableVariantOp(Operation *op);

/// Returns the StringAttr with the name `stringAttr` in the `targetAttr`, if
/// found.
std::optional<StringAttr> getConfigStringAttr(
    IREE::HAL::ExecutableTargetAttr targetAttr, StringRef stringAttr);

/// Returns the IntegerAttr with the name `integerAttr` in the `targetAttr`, if
/// found.
std::optional<IntegerAttr> getConfigIntegerAttr(
    IREE::HAL::ExecutableTargetAttr targetAttr, StringRef integerAttr);

/// Returns the BoolAttr with the name `integerAttr` in the `targetAttr`, if
/// found.
std::optional<BoolAttr> getConfigBoolAttr(
    IREE::HAL::ExecutableTargetAttr targetAttr, StringRef integerAttr);

/// Returns the LLVM Target triple associated with the `targetAttr`, if set.
std::optional<llvm::Triple> getTargetTriple(
    IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns the target architecture name, in IREE_ARCH convention, from the
/// given target triple.
const char *getIreeArchNameForTargetTriple(llvm::Triple triple);

/// Methods to get target information.
bool isVMVXBackend(IREE::HAL::ExecutableTargetAttr targetAttr);
bool hasMicrokernels(IREE::HAL::ExecutableTargetAttr targetAttr);

/// Checks if a tensor value is generated from a read-only object, like
/// and interface binding with read-only attribute or from an `arith.constant`
/// operation.
bool isReadOnly(Value v);

/// Return the static number of workgroup dispatched if it is known and
/// constant. Return an empty vector otherwise.
SmallVector<int64_t> getStaticNumWorkgroups(func::FuncOp funcOp);

//===----------------------------------------------------------------------===//
// Utility functions to set configurations
//===----------------------------------------------------------------------===//

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
  std::optional<int64_t> tileSize;
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
/// Returns the list of TilingInterface ops in the functions. If there are no
/// `scf.for` operations in the function return the TilingInterface operations
/// in the body of the function if it has a single basic block.
SmallVector<Operation *> getComputeOps(func::FuncOp funcOp);

/// If the given `forOp` is a tiled and distributed loop, returns its tiling and
/// distribution information.
std::optional<LoopTilingAndDistributionInfo> isTiledAndDistributedLoop(
    scf::ForOp forOp);

/// Collects information about loops matching tiled+distribute pattern.
SmallVector<LoopTilingAndDistributionInfo> getTiledAndDistributedLoopInfo(
    func::FuncOp funcOp);

Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes = {});

/// Returns the option that distributes the ops using the flow workgroup
/// ID/Count operations.
linalg::LinalgLoopDistributionOptions getIREELinalgLoopDistributionOptions(
    const SmallVector<int64_t> &tileSizes,
    linalg::DistributionMethod distributionMethod,
    int32_t maxWorkgroupParallelDims = kNumMaxParallelDims);

//===---------------------------------------------------------------------===//
// Misc. utility functions.
//===---------------------------------------------------------------------===//

/// Convert byte offset into offsets in terms of number of elements based
/// on `elementType`
OpFoldResult convertByteOffsetToElementOffset(RewriterBase &rewriter,
                                              Location loc,
                                              OpFoldResult byteOffset,
                                              Type elementType);

/// Replace the uses of memref value `origValue` with the given
/// `replacementValue`. Some uses of the memref value might require changes to
/// the operation itself. Create new operations which can carry the change, and
/// transitively replace their uses.
void replaceMemrefUsesAndPropagateType(RewriterBase &rewriter, Location loc,
                                       Value origValue, Value replacementValue);

/// Sink given operations as close as possible to their uses.
void sinkOpsInCFG(const SmallVector<Operation *> &allocs,
                  DominanceInfo &dominators);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_UTILS_UTILS_H_
