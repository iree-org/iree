// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- KernelDispatchUtils.cpp - Utilities for generating dispatch info ---===//
//
// This file defines utility functions that can be used to get the information
// about tile sizes to use to partition work across workgroups, the workgroup
// sizes and to create information the dispatch on the host side needs to
// execute an entry point function (e.g. total number of workgroups).
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

#define DEBUG_TYPE "kernel-dispatch-utils"

namespace mlir {
namespace iree_compiler {

/// Name of the StrAttr that can be used to get the key to access the tile size
/// information.
static const char kLaunchInfoKey[] = "launch_info_key";

/// Given `nprocs` try to distribute it evenly across 2 logical x and y.
static std::tuple<int64_t, int64_t> distributeProcs2D(int64_t nprocs) {
  int64_t nprocs_x = std::max<int64_t>(
      1, static_cast<int64_t>(
             llvm::PowerOf2Ceil(static_cast<uint64_t>(std::sqrt(nprocs)))));
  return std::make_tuple(nprocs_x, nprocs / nprocs_x);
}

/// For a given operation `op`, `options` and `resourceLimits` of the hardware
/// compute the
/// 1) number of tiling levels and tile sizes to use (updates `tileSizes`),
/// 2) workgroup size to use (updates `workgroupSize`),
/// 3) number of subgroups to use if two level tiling is used (updates
///    `numSubgroups`).
template <typename T>
static LogicalResult getOpLaunchConfig(T op, const SPIRVCodegenOptions &options,
                                       spirv::ResourceLimitsAttr resourceLimits,
                                       TileSizesListType &tileSizes,
                                       std::array<int64_t, 3> &workgroupSize,
                                       std::array<int64_t, 3> &numSubgroups) {
  return op.emitError("undefined launch config for tiled operation");
}

/// Launch config for `linalg.batchmatmul`.
template <>
LogicalResult getOpLaunchConfig(linalg::BatchMatmulOp op,
                                const SPIRVCodegenOptions &options,
                                spirv::ResourceLimitsAttr resourceLimits,
                                TileSizesListType &tileSizes,
                                std::array<int64_t, 3> &workgroupSize,
                                std::array<int64_t, 3> &numSubgroups) {
  unsigned maxWorkgroupSize =
      resourceLimits.max_compute_workgroup_invocations().getInt();
  std::tie(workgroupSize[0], workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);
  workgroupSize[2] = 1;
  // This is just being hard-wired for now to be minimal viable, but this can be
  // decided better when we have better estimates of device charecteristics.
  const int64_t nRowsPerWorkitem = 1;
  const int64_t nColsPerWorkitem = 1;
  const int64_t nBatchesPerWorkitem = 1;
  int64_t tileSizeK = 0;
  if (options.useWorkgroupMemory) {
    // This number should be decided based on the amount of shared memory
    // available (maybe). For now, just hard-wire it.
    tileSizeK = 32;
  }
  assert(tileSizes.empty());
  SmallVector<int64_t, 4> ts = {nBatchesPerWorkitem,
                                nRowsPerWorkitem * workgroupSize[1],
                                nColsPerWorkitem * workgroupSize[0], tileSizeK};
  tileSizes.emplace_back(std::move(ts));
  return success();
}

/// Returns the size of the co-operative matrix multiply operations on the
/// device.
static Optional<SmallVector<int64_t, 4>> getCooperativeMatmulSubgroupSize(
    spirv::ResourceLimitsAttr resourceLimits, Type lhsType, Type rhsType,
    Type initType, Type resultType) {
  for (auto coopMatmulProperties :
       resourceLimits.cooperative_matrix_properties_nv()
           .getAsRange<spirv::CooperativeMatrixPropertiesNVAttr>()) {
    if (coopMatmulProperties.a_type().getValue() == lhsType &&
        coopMatmulProperties.b_type().getValue() == rhsType &&
        coopMatmulProperties.c_type().getValue() == initType &&
        coopMatmulProperties.result_type().getValue() == resultType &&
        spirv::symbolizeScope(
            coopMatmulProperties.scope().getValue().getZExtValue())
                .getValue() == spirv::Scope::Subgroup) {
      return SmallVector<int64_t, 4>{
          coopMatmulProperties.m_size().getValue().getSExtValue(),
          coopMatmulProperties.n_size().getValue().getSExtValue(),
          coopMatmulProperties.k_size().getValue().getSExtValue()};
    }
  }
  return llvm::None;
}

/// Launch configuration for using spv.CooperativeMatrixMulAddNV
/// operations. Needs two levels of tiling.
static LogicalResult getConfigForCooperativeMatmul(
    linalg::MatmulOp op, const SPIRVCodegenOptions &options,
    spirv::ResourceLimitsAttr resourceLimits, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(op));
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix))
    return failure();

  ShapedType lhsType = op.inputs().front().getType().cast<ShapedType>();
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ShapedType rhsType = op.inputs().back().getType().cast<ShapedType>();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  ShapedType outputType =
      op.output_buffers().front().getType().cast<ShapedType>();

  Optional<SmallVector<int64_t, 4>> coopMatmulSize =
      getCooperativeMatmulSubgroupSize(
          resourceLimits, lhsType.getElementType(), rhsType.getElementType(),
          outputType.getElementType(), outputType.getElementType());
  if (!coopMatmulSize) return failure();

  // Check that the matmul sizes are a multiple of the tilesize.
  auto isMultipleOf = [](int64_t s, int64_t ts) {
    return !ShapedType::isDynamic(s) && (s % ts) == 0;
  };
  if (!isMultipleOf(lhsShape[0], (*coopMatmulSize)[0]) ||
      !isMultipleOf(rhsShape[1], (*coopMatmulSize)[1]) ||
      !isMultipleOf(lhsShape[1], (*coopMatmulSize)[2]) ||
      !isMultipleOf(rhsShape[0], (*coopMatmulSize)[2]))
    return failure();

  if (options.useWorkgroupMemory) {
    numSubgroups[0] = 2;
    numSubgroups[1] = 2;
  } else {
    numSubgroups[0] = 1;
    numSubgroups[1] = 1;
  }
  numSubgroups[2] = 1;

  // For now this is being hard-wired to be {4, 4, 2}. This can actually be set
  // to whatever, but ultimately depends on register pressure.
  const int64_t numVecMatmulPerSubgroupX = 4;
  const int64_t numVecMatmulPerSubgroupY = 4;
  const int64_t numVecMatmulPerSubgroupK = 2;
  SmallVector<int64_t, 4> ts = {
      numVecMatmulPerSubgroupY * (*coopMatmulSize)[0] * numSubgroups[1],
      numVecMatmulPerSubgroupX * (*coopMatmulSize)[1] * numSubgroups[0],
      numVecMatmulPerSubgroupK * (*coopMatmulSize)[2]};
  tileSizes.emplace_back(std::move(ts));

  int64_t subgroupSize =
      resourceLimits.subgroup_size().getValue().getSExtValue();
  workgroupSize[0] = numSubgroups[0] * numSubgroups[1] * subgroupSize;
  workgroupSize[1] = 1;
  workgroupSize[2] = 1;
  // Subgroup tile sizes
  SmallVector<int64_t, 4> subgroupTs = {
      numVecMatmulPerSubgroupY * (*coopMatmulSize)[0],
      numVecMatmulPerSubgroupX * (*coopMatmulSize)[1]};
  tileSizes.emplace_back(std::move(subgroupTs));
  return success();
}

/// Launch configuration for different known GPU configuration.
static LogicalResult getTargetSpecificConfig(
    linalg::MatmulOp op, const SPIRVCodegenOptions &options,
    spirv::ResourceLimitsAttr resourceLimits, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  if (spirv::lookupTargetEnv(op).getVendorID() != spirv::Vendor::ARM)
    return failure();
  workgroupSize[0] = resourceLimits.subgroup_size().getInt();
  workgroupSize[1] = 1;
  workgroupSize[2] = 1;
  SmallVector<int64_t, 4> ts = {8, 64, 4};
  tileSizes.emplace_back(ts);
  // No tiling at the subgroup level since this target doesn't use subgroup op
  // or shared memory.
  tileSizes.emplace_back();
  SmallVector<int64_t, 4> threadTs = {ts[0], ts[1] / workgroupSize[0], ts[2]};
  tileSizes.emplace_back(threadTs);
  return success();
}

template <>
LogicalResult getOpLaunchConfig(linalg::MatmulOp op,
                                const SPIRVCodegenOptions &options,
                                spirv::ResourceLimitsAttr resourceLimits,
                                TileSizesListType &tileSizes,
                                std::array<int64_t, 3> &workgroupSize,
                                std::array<int64_t, 3> &numSubgroups) {
  if (options.useVectorization && succeeded(getConfigForCooperativeMatmul(
                                      op, options, resourceLimits, tileSizes,
                                      workgroupSize, numSubgroups))) {
    return success();
  } else if (options.useVectorization &&
             succeeded(getTargetSpecificConfig(op, options, resourceLimits,
                                               tileSizes, workgroupSize,
                                               numSubgroups))) {
    return success();
  }
  unsigned maxWorkgroupSize =
      resourceLimits.max_compute_workgroup_invocations().getInt();
  std::tie(workgroupSize[0], workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);
  workgroupSize[2] = 1;
  const int nRowsPerWorkitem = 1;
  const int nColsPerWorkitem = 1;
  int64_t tileSizeK = 0;
  if (options.useWorkgroupMemory) {
    // TODO(#3131): This number should be decided based on the amount of shared
    // memory available (maybe). For now, just hard-wire it.
    tileSizeK = 32;
  }
  assert(tileSizes.empty());
  SmallVector<int64_t, 4> ts = {nRowsPerWorkitem * workgroupSize[1],
                                nColsPerWorkitem * workgroupSize[0], tileSizeK};
  tileSizes.emplace_back(std::move(ts));
  return success();
}

template <>
LogicalResult getOpLaunchConfig(linalg::ConvOp op,
                                const SPIRVCodegenOptions &options,
                                spirv::ResourceLimitsAttr resourceLimits,
                                TileSizesListType &tileSizes,
                                std::array<int64_t, 3> &workgroupSize,
                                std::array<int64_t, 3> &numSubgroups) {
  unsigned maxWorkgroupSize =
      resourceLimits.max_compute_workgroup_invocations().getInt();
  const int64_t tileSizeX = 32;
  int64_t tileSizeY = maxWorkgroupSize / tileSizeX;
  SmallVector<int64_t, 4> ts = {1, tileSizeY, tileSizeX};
  tileSizes.emplace_back(std::move(ts));
  workgroupSize = {tileSizeX, tileSizeY, 1};
  return success();
}

template <typename PoolingOpTy>
static LogicalResult getPoolingOpLaunchConfig(
    PoolingOpTy op, const SPIRVCodegenOptions &options,
    spirv::ResourceLimitsAttr resourceLimits, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  unsigned maxWorkgroupSize =
      resourceLimits.max_compute_workgroup_invocations().getInt();
  // Pooling op seems to be rank polymorphic but is not well specified enough to
  // be able to figure out which dimensions of the output correspond to the
  // pooled dimension and which are not. Need to fix that, but for now just use
  // a working heuristic.
  SmallVector<int64_t, 4> ts(std::min<int64_t>(
      op.output().getType().template cast<ShapedType>().getRank(), 3));
  const int64_t tileSizeX = 32;
  int64_t tileSizeY = maxWorkgroupSize / tileSizeX;
  ts[ts.size() - 2] = tileSizeY;
  ts[ts.size() - 1] = tileSizeX;
  tileSizes.emplace_back(std::move(ts));
  workgroupSize = {tileSizeX, tileSizeY, 1};
  return success();
}

#define DEFINE_POOLING_OP_CONFIG(opName)                                      \
  template <>                                                                 \
  LogicalResult getOpLaunchConfig(                                            \
      opName op, const SPIRVCodegenOptions &options,                          \
      spirv::ResourceLimitsAttr resourceLimits, TileSizesListType &tileSizes, \
      std::array<int64_t, 3> &workgroupSize,                                  \
      std::array<int64_t, 3> &numSubgroups) {                                 \
    return getPoolingOpLaunchConfig(op, options, resourceLimits, tileSizes,   \
                                    workgroupSize, numSubgroups);             \
  }

DEFINE_POOLING_OP_CONFIG(linalg::PoolingMaxOp)
DEFINE_POOLING_OP_CONFIG(linalg::PoolingMinOp)
DEFINE_POOLING_OP_CONFIG(linalg::PoolingSumOp)

#undef DEFINE_POOLINGOP_CONFIG

Optional<StringRef> LaunchConfig::getKey(Operation *op) const {
  StringAttr attr = op->getAttrOfType<StringAttr>(kLaunchInfoKey);
  if (!attr) return {};
  return attr.getValue();
}

LogicalResult LaunchConfig::init(MLIRContext *context,
                                 const SPIRVCodegenOptions &options,
                                 ArrayRef<Operation *> linalgOps) {
  unsigned numTiledOps = 0;
  auto setKey = [&](Operation *op) -> std::string {
    std::string key = llvm::formatv("__op_num_{0}__", numTiledOps++).str();
    op->setAttr(Identifier::get(kLaunchInfoKey, context),
                StringAttr::get(key, context));
    return key;
  };

  if (!options.workgroupSize.empty()) {
    for (Operation *linalgOp : linalgOps)
      tileSizes[setKey(linalgOp)].emplace_back(options.tileSizes.begin(),
                                               options.tileSizes.end());
    workgroupSize = {1, 1, 1};
    for (unsigned i = 0,
                  e = std::min<unsigned>(3, options.workgroupSize.size());
         i != e; ++i)
      workgroupSize[i] = options.workgroupSize[i];
    return success();
  }

  if (linalgOps.empty()) return success();

  spirv::ResourceLimitsAttr resourceLimits =
      spirv::lookupTargetEnv(*linalgOps.begin()).getResourceLimits();

  Optional<linalg::LinalgOp> rootOperation = {};

  for (Operation *op : linalgOps) {
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
#define DISPATCH(opName)                                                      \
  if (auto lOp = dyn_cast<opName>(linalgOp.getOperation())) {                 \
    if (rootOperation) {                                                      \
      return lOp.emitError(                                                   \
          "unhandled multiple root operations in dispatch region");           \
    }                                                                         \
    rootOperation = cast<linalg::LinalgOp>(lOp.getOperation());               \
    TileSizesListType &tileSizesInfo = tileSizes[setKey(*rootOperation)];     \
    if (failed(getOpLaunchConfig(lOp, options, resourceLimits, tileSizesInfo, \
                                 workgroupSize, numSubgroups))) {             \
      return failure();                                                       \
    }                                                                         \
    continue;                                                                 \
  }

    DISPATCH(linalg::BatchMatmulOp)
    DISPATCH(linalg::ConvOp)
    DISPATCH(linalg::MatmulOp)
    DISPATCH(linalg::PoolingMaxOp)
    DISPATCH(linalg::PoolingMinOp)
    DISPATCH(linalg::PoolingSumOp)

#undef DISPATCH
  }

  // TODO(ravishankarm): Verify that the set configurations is within the device
  // limits.
  return success();
}

void LaunchConfig::finalize(FuncOp funcOp) {
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    linalgOp.removeAttr(Identifier::get(kLaunchInfoKey, funcOp.getContext()));
  });
}

template <typename OpTy>
static Optional<SmallVector<int64_t, 4>> getOpNativeVectorSize(OpTy op) {
  return llvm::None;
}

template <>
Optional<SmallVector<int64_t, 4>> getOpNativeVectorSize<vector::ContractionOp>(
    vector::ContractionOp op) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(op));
  if (targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
      targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    spirv::ResourceLimitsAttr resourceLimits =
        spirv::lookupTargetEnv(op).getResourceLimits();
    return getCooperativeMatmulSubgroupSize(
        resourceLimits, op.getLhsType().getElementType(),
        op.getRhsType().getElementType(),
        op.getAccType().cast<VectorType>().getElementType(),
        op.getResultType().cast<VectorType>().getElementType());
  } else {
    // Map to vec4 fma operations.
    return SmallVector<int64_t, 4>({1, 4, 1});
  }
}

template <>
Optional<SmallVector<int64_t, 4>> getOpNativeVectorSize<vector::TransferReadOp>(
    vector::TransferReadOp op) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(op));
  if (targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
      targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    // Don't unroll cooperative martrix load as they should match the size of
    // the contract.
    return SmallVector<int64_t, 4>(op.getVectorType().getDimSize(0),
                                   op.getVectorType().getDimSize(1));
  } else {
    // Map to load4.
    return SmallVector<int64_t, 4>({1, 4});
  }
}

Optional<SmallVector<int64_t, 4>> getNativeVectorSize(Operation *op) {
#define DISPATCH(opname)                            \
  if (isa<opname>(op)) {                            \
    return getOpNativeVectorSize(cast<opname>(op)); \
  }

  DISPATCH(vector::ContractionOp)
  DISPATCH(vector::TransferReadOp)

#undef DISPATCH
  return llvm::None;
}

}  // namespace iree_compiler
}  // namespace mlir
