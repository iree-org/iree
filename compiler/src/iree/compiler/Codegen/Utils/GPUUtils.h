// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_GPUUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_GPUUTILS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

static constexpr int32_t kNumGPUDims = 3;
static constexpr int32_t kWarpSize = 32;

//===----------------------------------------------------------------------===//
// GPU processor IDs and sizes
//===----------------------------------------------------------------------===//

llvm::SmallVector<linalg::ProcInfo, 2>
getGPUThreadIdsAndCounts(OpBuilder &builder, Location loc, unsigned numDims,
                         llvm::ArrayRef<int64_t> workgroupSize);

/// Computes subgroup ID and returns in (X, Y, Z) order.
///
/// Note that CUDA doesn't have a subgroupId equivalent so we are are computing
/// the subgroup ID based on the threadID. When tiling to warp we assume each
/// warp is full and we pick a workgroup size so that `workgroupSize.x %
/// warpSize == 0`. This is why we can have warpId = { threadId.x / warpSize,
/// threadId.y, threadId.z }.
llvm::SmallVector<linalg::ProcInfo, 2>
getSubgroupIdsAndCounts(OpBuilder &builder, Location loc, unsigned warpSize,
                        unsigned numDims, llvm::ArrayRef<int64_t> numSubgroups);

/// Indicates whether the given array of DeviceMappingAttrInterfaces is a
/// descending relative mapping, for example:
///  [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>]
/// or
///  [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
bool isDescendingRelativeMappingIndices(ArrayRef<Attribute> array);

// Indicates whether the given `scf.forall` op has a processor ID mapping of
// the template type(s).
template <typename... Type>
bool forallOpHasMappingType(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping || mapping.value().empty()) {
    return false;
  }

  return isa<Type...>(*mapping.value().begin());
}

// Indicates whether an operation is within a distributed context with the
// specified mapping type(s).
template <typename... Type>
bool operationHasParentForallOfMappingType(Operation *op) {
  auto parentForallOp = op->getParentOfType<scf::ForallOp>();
  while (parentForallOp) {
    if (forallOpHasMappingType<Type...>(parentForallOp)) {
      return true;
    }
    parentForallOp = parentForallOp->getParentOfType<scf::ForallOp>();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// GPU vectorization
//===----------------------------------------------------------------------===//

/// Returns true if we can use all threads to perform vectorized load/store of
/// the given `shape`.
bool canPerformVectorAccessUsingAllThreads(ArrayRef<int64_t> shape,
                                           int64_t threadCount,
                                           int64_t vectorSize);

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register. This is needed to get good performance on sm_80 target.
std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract);

//===----------------------------------------------------------------------===//
// GPU tiling and distribution
//===----------------------------------------------------------------------===//

/// Returns the attribute name carrying information about distribution.
const char *getGPUDistributeAttrName();

/// Returns the tile sizes at the given `tilingLevel` for compute ops in
/// `funcOp`.
FailureOr<SmallVector<int64_t>> getGPUTileSize(mlir::FunctionOpInterface funcOp,
                                               int tilingLevel);

/// Returns the functor to compute tile sizes at the given `tilingLevel` for
/// compute ops in `funcOp`.
FailureOr<scf::SCFTileSizeComputationFunction>
getGPUScfTileSizeComputeFn(mlir::FunctionOpInterface funcOp, int tilingLevel);

//===----------------------------------------------------------------------===//
// GPU workgroup memory
//===----------------------------------------------------------------------===//

/// Allocates GPU workgroup memory matching the given `subview`. If there are
/// dynamic dimensions, the bounds are in `sizeBounds`.
std::optional<Value> allocateWorkgroupMemory(OpBuilder &builder,
                                             memref::SubViewOp subview,
                                             ArrayRef<Value> sizeBounds,
                                             DataLayout &);

/// Deallocates GPU workgroup memory behind `buffer`.
LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value buffer);

/// Copies `src` value to `dst` in shared memory.
LogicalResult copyToWorkgroupMemory(OpBuilder &builder, Value src, Value dst);

/// Propagates shared memory copy to producer linalg.fill or consumer
/// linalg.generic when possible.
void propagateSharedMemoryCopy(mlir::FunctionOpInterface funcOp);

/// Inserts barriers before and after shared memory copy.
void insertBarriersAroundSharedMemoryCopy(mlir::FunctionOpInterface funcOp);

/// Emit reduction across a group for a given input. Emits `gpu.shuffle`
/// based reduction only when `expandSubgroupReduce` is set.
Value emitGPUGroupReduction(Location loc, OpBuilder &builder, Value input,
                            vector::CombiningKind kind, uint32_t size,
                            int warpSize, bool expandSubgroupReduce);

/// Return the native size of an operation used in contraction calculation.
// TODO: Make this take HW specific sizes.
std::optional<SmallVector<int64_t>> getWmmaNativeVectorSize(Operation *op);

/// Helper function to return native size for MMA.SYNC-based operations.
std::optional<SmallVector<int64_t>> getMmaNativeVectorSize(Operation *op);

/// Return true if the given memref has workgroup memory space.
bool hasSharedMemoryAddressSpace(MemRefType memrefType);

/// Packs vector of lower precision into a single 32-bit width element.
/// (i.e <2xf16> -> i32 and <4xi8> -> i32)
Value packVectorToSupportedWidth(Location loc, OpBuilder &builder, Value input);

/// Unpack single scalar element into a target vector type.
/// (i.e i32 -> vector<4xi8> or f32 -> vector<2xf16>)
Value unpackToVector(Location loc, OpBuilder &builder, Value packedInput,
                     VectorType targetVecType);

/// Emit identity constant based on combiningKind and type.
Value getCombiningIdentityValue(Location loc, OpBuilder &builder,
                                vector::CombiningKind kind, Type identityType);
//===----------------------------------------------------------------------===//
// GPU CodeGen op filter
//===----------------------------------------------------------------------===//

/// Returns true if the index map represents a transpose that benefits from
/// using shared memory when CodeGen towards the GPU.
bool sharedMemTransposeFilter(AffineMap indexMap);

//===----------------------------------------------------------------------===//
// GPU UKernel Utils
//===----------------------------------------------------------------------===//

/// Checks if target Chip(StringRef) has UKernel support.
bool hasUkernelSupportedRocmArch(StringRef targetChip);

/// Checks if targetAttr's GPU target has UKernel support.
bool hasUkernelSupportedGpuArch(IREE::HAL::ExecutableTargetAttr targetAttr);

//===----------------------------------------------------------------------===//
// GPU Target Information
//===----------------------------------------------------------------------===//
FailureOr<ArrayAttr> getSupportedMmaTypes(DictionaryAttr config);

FailureOr<ArrayAttr> getSupportedMmaTypes(mlir::FunctionOpInterface entryPoint);

/// Returns the GPU target attribute from `iree-gpu-test-target` if provided.
/// Returns null TargetAttr othersise.
IREE::GPU::TargetAttr getCLGPUTarget(MLIRContext *context);

/// Returns the GPU target attribute from executable |target| if found.
/// Returns null TargetAttr othersise.
IREE::GPU::TargetAttr getGPUTargetAttr(IREE::HAL::ExecutableTargetAttr target);
/// Returns the GPU target attribute from the executable target wrapping |op|
/// if found. Returns null TargetAttr othersise.
IREE::GPU::TargetAttr getGPUTargetAttr(Operation *op);

/// Returns the GPU subgroup size chosen for the current CodeGen pipeline if
/// exists; otherwise returns the subgroup size from the GPU target description.
/// Returns std::nullopt if none found.
std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_GPUUTILS_H_
