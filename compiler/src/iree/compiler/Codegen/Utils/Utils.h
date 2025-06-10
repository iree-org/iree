// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_UTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_UTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SubsetOpInterface.h"

namespace mlir::iree_compiler {

static constexpr unsigned kNumMaxParallelDims = 3;

//===----------------------------------------------------------------------===//
// Utility functions to get entry points
//===----------------------------------------------------------------------===//

/// Returns true if the given `func` is a kernel dispatch entry point.
bool isEntryPoint(mlir::FunctionOpInterface func);

/// Returns the entry point op for the `funcOp`. Returns `nullptr` on failure.
std::optional<IREE::HAL::ExecutableExportOp>
getEntryPoint(mlir::FunctionOpInterface funcOp);

/// Returns the StringAttr with the name `stringAttr` in the `srcAttr`, if
/// found. The `srcAttr` can be either IREE::HAL::ExecutableTargetAttr or
/// DictionaryAttr.
std::optional<StringAttr> getConfigStringAttr(Attribute srcAttr,
                                              StringRef stringAttr);

/// Returns the IntegerAttr with the name `integerAttr` in the `srcAttr`, if
/// found.
std::optional<IntegerAttr> getConfigIntegerAttr(Attribute srcAttr,
                                                StringRef integerAttr);

/// Returns the BoolAttr with the name `boolAttr` in the `srcAttr`, if
/// found.
std::optional<BoolAttr> getConfigBoolAttr(Attribute srcAttr,
                                          StringRef boolAttr);

/// Returns the LLVM Target triple associated with the `attr`, if set.
std::optional<llvm::Triple> getTargetTriple(Attribute attr);

/// Returns the target architecture name, in IREE_ARCH convention, from the
/// given target triple.
const char *getIreeArchNameForTargetTriple(llvm::Triple triple);

/// Methods to get target information.
bool isLLVMCPUBackend(IREE::HAL::ExecutableTargetAttr targetAttr);
bool isVMVXBackend(IREE::HAL::ExecutableTargetAttr targetAttr);
bool isROCMBackend(IREE::HAL::ExecutableTargetAttr targetAttr);
bool isWebGPUBackend(IREE::HAL::ExecutableTargetAttr targetAttr);

// Returns true if the ukernel with given `ukernelName` is enabled.
// If `ukernelName` is empty (the default), returns true if any ukernel
// is enabled at all.
bool hasUkernel(Attribute attr, StringRef ukernelName = "");

/// Returns the CPU target features associated with the `attr`, if found.
std::optional<StringRef> getCpuFeatures(Attribute attr);

/// Returns true if `attr` has `feature` in its CPU features.
bool hasFeature(Attribute attr, StringRef feature);

/// Architecture identification.
bool isX86(Attribute attr);
bool isX86_64(Attribute attr);
bool isAArch64(Attribute attr);
bool isRISCV(Attribute attr);
bool isRISCV32(Attribute attr);

/// Checks if a tensor value is generated from a read-only object, like
/// and interface binding with read-only attribute or from an `arith.constant`
/// operation.
bool isReadOnly(Value v);

/// Multiple uses of `tensor.empty()` results in a copy since upstream
/// treats `tensor.empty()` as an allocation and sees uses as a data-hazard
/// creating copies/allocations. Since the `empty` op is a proxy for
/// undef, these could just be duplicated to have a single use. This removes
/// unnecessary data-hazards.
LogicalResult duplicateTensorEmptyOps(OpBuilder &b, tensor::EmptyOp emptyOp);

/// Return the static number of workgroup dispatched if it is known and
/// constant. If it is not known, it will return ShapedType::kDynamic.
SmallVector<int64_t> getStaticNumWorkgroups(mlir::FunctionOpInterface funcOp);

//===----------------------------------------------------------------------===//
// Utility functions to set configurations
//===----------------------------------------------------------------------===//

LogicalResult setDefaultCustomOpLoweringConfig(
    mlir::FunctionOpInterface FunctionOpInterface,
    IREE::LinalgExt::CustomOp customOp,
    std::function<LogicalResult(mlir::FunctionOpInterface)> configFn);

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

/// Returns the list of TilingInterface ops in the operation obtained by a
/// post order walk of the operation. This implies that in case of
/// nested compute ops, the outermost compute ops are towards the end of the
/// list.
SmallVector<Operation *> getComputeOps(Operation *containingOp);

/// If the given `forOp` is a tiled and distributed loop, returns its tiling and
/// distribution information.
std::optional<LoopTilingAndDistributionInfo>
isTiledAndDistributedLoop(scf::ForOp forOp);

/// Collects information about loops matching tiled+distribute pattern.
SmallVector<LoopTilingAndDistributionInfo>
getTiledAndDistributedLoopInfo(mlir::FunctionOpInterface funcOp);

/// Sets the tile sizes of the SCFTilingOptions. If `tileScalableFlags` are
/// provided the corresponding tile size will be multiplied by a vector.vscale
/// op.
void setSCFTileSizes(scf::SCFTilingOptions &options, TilingInterface op,
                     ArrayRef<int64_t> tileSizes,
                     ArrayRef<bool> tileScalableFlags);

Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes = {});

/// Returns the option that distributes the ops using the flow workgroup
/// ID/Count operations.
linalg::LinalgLoopDistributionOptions getIREELinalgLoopDistributionOptions(
    linalg::DistributionMethod distributionMethod,
    int32_t maxWorkgroupParallelDims = kNumMaxParallelDims);

// Helper to construct the strategy attribute dictionary for a pipeline that
// does software pipelining.
DictionaryAttr
getSoftwarePipeliningAttrDict(MLIRContext *context,
                              unsigned softwarePipelineDepth = 0,
                              unsigned softwarePipelineStoreStage = 1);

// Helpers to extract the pipelining configuration from the config dictionary.
FailureOr<int64_t> getSoftwarePipelineDepth(DictionaryAttr);
FailureOr<int64_t> getSoftwarePipelineStoreStage(DictionaryAttr);

// Returns a small tiling factor for the given reduction `dimSize`.
// Returns 0 to avoid tiling.
int getReductionTilingFactor(int64_t dimSize);

// Returns the minimal element bitwidth used in the operands and results of the
// given Linalg op.
int64_t getMinElementBitwidth(linalg::LinalgOp linalgOp);

//===---------------------------------------------------------------------===//
// Bufferization utility functions
//===---------------------------------------------------------------------===//

/// Find the memref version of the given InterfaceBindingSubspanOp. If no such
/// op exists in the same block (before the given op), create a new op.
Value findOrCreateSubspanBuffer(RewriterBase &rewriter,
                                IREE::HAL::InterfaceBindingSubspanOp subspanOp);

//===---------------------------------------------------------------------===//
// Misc. utility functions.
//===---------------------------------------------------------------------===//

/// Given a list of `Value`s, set the insertion point to the last (least
/// dominant) of these values.
Operation *setInsertionPointAfterLastValue(OpBuilder &builder,
                                           ArrayRef<Value> values);

/// Given a SubsetInsertionOpInterface, find all values that are needed to
/// build an equivalent subset extraction, and set the insertion point to the
/// last of these values.
Operation *
setInsertionPointAfterLastNeededValue(OpBuilder &builder,
                                      SubsetInsertionOpInterface subsetOp);

/// Moves the op to right after its last (most dominant) operand. If the operand
/// is a block argument, then the op is moved to the start of the block.
void moveOpAfterLastOperand(RewriterBase &rewriter, DominanceInfo &domInfo,
                            Operation *op);

/// Check if the two tensor types (with their respective dynamic dimension
/// values) have the same shape.
bool equalTensorShape(RankedTensorType tensorType, ValueRange tensorDynSizes,
                      IREE::TensorExt::DispatchTensorType dispatchTensorType,
                      ValueRange dispatchTensorDynSizes);

/// Convert byte offset into offsets in terms of number of elements based
/// on `elementType`
OpFoldResult convertByteOffsetToElementOffset(RewriterBase &rewriter,
                                              Location loc,
                                              OpFoldResult byteOffset,
                                              Type elementType);

/// Clone an operation and drop all encodings.
Operation *dropEncodingAndCloneOp(OpBuilder &builder, Operation *op,
                                  ValueRange convertedInputOperands,
                                  ValueRange convertedOutputOperands);

/// Replace the uses of memref value `origValue` with the given
/// `replacementValue`. Some uses of the memref value might require changes to
/// the operation itself. Create new operations which can carry the change, and
/// transitively replace their uses.
void replaceMemrefUsesAndPropagateType(RewriterBase &rewriter, Location loc,
                                       Value origValue, Value replacementValue);

/// Sink given operations as close as possible to their uses.
void sinkOpsInCFG(const SmallVector<Operation *> &allocs,
                  DominanceInfo &dominators);

// Check if there is a fused linalg op present in the backward slice of any of
// the inputs.
bool hasFusedLeadingOp(linalg::LinalgOp rootOp);

std::optional<vector::VscaleRange>
getDefaultVscaleRange(IREE::HAL::ExecutableTargetAttr targetAttr);

using DimBound = vector::ConstantOrScalableBound;
using DimBoundSize = DimBound::BoundSize;

/// Should the scalable upper bound be rounded up to the nearest multiple of
/// vscale?
enum class RoundUpVscaleMultiple { No, Yes };

/// Computes the upper bound of `dimNum` dim of the ShapedType value
/// `shapedValue`. If the optional `vscaleRange` is provided then the computed
/// bound can be a scalable quantity.
FailureOr<DimBoundSize>
computeDimUpperBound(Value shapedValue, unsigned dimNum,
                     std::optional<vector::VscaleRange> vscaleRange,
                     RoundUpVscaleMultiple = RoundUpVscaleMultiple::No);

// Utility to make sure we are storing the full incoming subspan. Otherwise we
// cannot simply adjust the subspan's resultant type later.
bool isFullSlice(OffsetSizeAndStrideOpInterface sliceLoadStoreOp,
                 IREE::TensorExt::DispatchTensorType tensorType,
                 ValueRange dynamicDims);

//===----------------------------------------------------------------------===//
// Utility functions for vector size inference for dynamic shapes
//===----------------------------------------------------------------------===//

struct VectorizationTileSizes {
  SmallVector<int64_t> destShape;
  SmallVector<int64_t> vectorSizes;
  SmallVector<bool> vectorScalableFlags;
};

/// Returns a VectorizationTileSizes which contains the inferred bounded result
/// shape and vector input sizes. This is useful to infer the sizes from a
/// chain.
std::optional<VectorizationTileSizes> inferSizesFromIR(Value val);

/// Returns the result sizes and vector input sizes of the linalg.unpack op. The
/// inferred bounding size is returned if it is dynamic shape. Returns
/// std::nullopt if the shape inference failed.
std::optional<VectorizationTileSizes> inferSizesFromIR(linalg::UnPackOp op);

/// Returns the result sizes and vector input sizes of the linalg.pack op. The
/// inferred bounding size is returned if it is dynamic shape. Returns
/// std::nullopt if the shape inference failed.
std::optional<VectorizationTileSizes> inferSizesFromIR(linalg::PackOp op);

/// Tries to infer the vector sizes from an IR using ValueBounds analysis. If
/// `opResult` is provided, it stores the bounded result shapes to destShape.
/// Returns std::nullopt if vector sizes can't be inferred.
std::optional<VectorizationTileSizes>
inferSizesFromIR(linalg::LinalgOp linalgOp, std::optional<OpResult> opResult);

/// Returns the underlying index if the given value is a constant index.
std::optional<int64_t> getConstantIndex(Value value);

/// Return true if we can prove that the we always run at least the first
/// iteration of the ForOp.
bool alwaysRunsFirstIteration(scf::ForOp op);

/// Return true if we can prove that the we never run more than one iteration of
/// the ForOp.
bool neverRunsSecondIteration(scf::ForOp op);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_UTILS_H_
