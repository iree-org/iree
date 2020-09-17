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
#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

#define DEBUG_TYPE "kernel-dispatch-utils"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Number of workgroups computation
//===----------------------------------------------------------------------===//

FuncOp getNumWorkgroupsFn(FuncOp entryPointFn) {
  SymbolRefAttr attr =
      entryPointFn.getAttrOfType<SymbolRefAttr>(getNumWorkgroupsFnAttrName());
  if (!attr) {
    entryPointFn.emitError("missing attribute '")
        << getNumWorkgroupsFnAttrName() << "'";
    return nullptr;
  }
  FuncOp numWorkgroupsFn = dyn_cast_or_null<FuncOp>(SymbolTable::lookupSymbolIn(
      entryPointFn.getParentOfType<ModuleOp>(), attr));
  if (!numWorkgroupsFn) {
    entryPointFn.emitError("unable to find num workgroups fn ") << attr;
    return nullptr;
  }
  if (!numWorkgroupsFn.empty()) {
    entryPointFn.emitError("num workgroups fn expected to be empty");
    return nullptr;
  }
  return numWorkgroupsFn;
}

/// Computes the bounds of the parallel loops partitioned across workgroups.
static Optional<SmallVector<Value, 2>> getParallelLoopRange(
    PatternRewriter &rewriter, Location loc, linalg::LinalgOp linalgOp) {
  FuncOp numWorkgroupsFn =
      getNumWorkgroupsFn(linalgOp.getParentOfType<FuncOp>());
  if (!numWorkgroupsFn) return {};
  LLVM_DEBUG({
    llvm::dbgs() << "Found num workgroups function : "
                 << numWorkgroupsFn.getName();
  });
  rewriter.setInsertionPointToEnd(numWorkgroupsFn.addEntryBlock());
  llvm::SetVector<Operation *> slice;
  getBackwardSlice(linalgOp, &slice);
  BlockAndValueMapping mapper;
  for (Operation *op : slice) {
    rewriter.clone(*op, mapper);
  }
  // Clone the linalg operation just to compute the loop bounds.
  linalg::LinalgOp clonedLinalgOp =
      rewriter.clone(*linalgOp.getOperation(), mapper);
  Optional<SmallVector<Value, 4>> bounds =
      getLoopRanges(rewriter, clonedLinalgOp);
  unsigned numParallelLoops = linalgOp.iterator_types()
                                  .getValue()
                                  .take_while([](Attribute attr) -> bool {
                                    return attr.cast<StringAttr>().getValue() ==
                                           getParallelIteratorTypeName();
                                  })
                                  .size();
  SmallVector<Value, 2> returnVals(
      bounds->begin(), std::next(bounds->begin(), numParallelLoops));
  rewriter.eraseOp(clonedLinalgOp);
  return returnVals;
}

/// Utility method to build IR that computes ceil(`numerator` / `denominator`)
static Value buildCeilDiv(PatternRewriter &rewriter, Location loc,
                          Value numerator, Value denominator) {
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  Value t = rewriter.create<AddIOp>(
      loc, numerator, rewriter.create<SubIOp>(loc, denominator, one));
  return rewriter.create<SignedDivIOp>(loc, t, denominator);
}

/// Utility method to build IR that computes ceil(`numerator` / `denominator`)
/// when denominator is a constant.
static Value buildCeilDivConstDenominator(PatternRewriter &rewriter,
                                          Location loc, Value numerator,
                                          int64_t denominator) {
  return buildCeilDiv(rewriter, loc, numerator,
                      rewriter.create<ConstantIndexOp>(loc, denominator));
}

LogicalResult createNumWorkgroupsFromResultShape(PatternRewriter &rewriter,
                                                 linalg::LinalgOp linalgOp,
                                                 FuncOp entryPointFn,
                                                 ArrayRef<int64_t> tileSizes) {
  Location loc = linalgOp.getLoc();
  OpBuilder::InsertionGuard gaurd(rewriter);
  Optional<SmallVector<Value, 2>> parallelLoopRange =
      getParallelLoopRange(rewriter, loc, linalgOp);
  if (!parallelLoopRange) return failure();
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  SmallVector<Value, 3> returnValues(3, one);
  for (size_t i = 0, e = std::min<size_t>(parallelLoopRange->size(), 3); i != e;
       ++i) {
    returnValues[i] = buildCeilDivConstDenominator(
        rewriter, loc, (*parallelLoopRange)[e - i - 1], tileSizes[e - i - 1]);
  }
  rewriter.create<mlir::ReturnOp>(loc, returnValues);
  return success();
}

LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    int64_t workgroupSizeX) {
  Location loc = linalgOp.getLoc();
  OpBuilder::InsertionGuard gaurd(rewriter);
  Optional<SmallVector<Value, 2>> parallelLoopRange =
      getParallelLoopRange(rewriter, loc, linalgOp);
  if (!parallelLoopRange) return failure();
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  SmallVector<Value, 3> returnValues(3, one);
  for (auto range : *parallelLoopRange) {
    returnValues[0] = rewriter.create<MulIOp>(loc, range, returnValues[0]);
  }
  returnValues[0] = buildCeilDivConstDenominator(rewriter, loc, returnValues[0],
                                                 workgroupSizeX);
  rewriter.create<mlir::ReturnOp>(loc, returnValues);
  return success();
}

//===----------------------------------------------------------------------===//
// Launch config calculation.
//===----------------------------------------------------------------------===//

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
  // TODO(#3131): This is just being hard-wired for now to be minimal viable,
  // but this can be decided better when we have better estimates of device
  // charecteristics.
  const int64_t nRowsPerWorkitem = 1;
  const int64_t nColsPerWorkitem = 1;
  const int64_t nBatchesPerWorkitem = 1;
  int64_t tileSizeK = 0;
  if (options.useWorkgroupMemory) {
    // TODO(#3131): This number should be decided based on the amount of
    // shared memory available (maybe). For now, just hard-wire it.
    tileSizeK = 32;
  }
  assert(tileSizes.empty());
  SmallVector<int64_t, 4> ts = {nBatchesPerWorkitem,
                                nRowsPerWorkitem * workgroupSize[1],
                                nColsPerWorkitem * workgroupSize[0], tileSizeK};
  tileSizes.emplace_back(std::move(ts));
  return success();
}

/// The size of the co-operative matrix multiply operations on the device.
// TODO(#3131): This needs to be queried from the device.
Optional<std::array<int64_t, 3>> getCooperativeMatmulSubgroupSize(
    Type dataType, Type accumulatorType) {
  if (dataType.isInteger(8) && accumulatorType.isInteger(32)) {
    return std::array<int64_t, 3>{8, 8, 32};
  }
  if (dataType.isF16() &&
      (accumulatorType.isF32() || accumulatorType.isF16())) {
    return std::array<int64_t, 3>{8, 8, 16};
  }
  return {};
}

/// Launch configuration for using spv.CooperativeMatrixMulAddNV
/// operations. Needs two levels of tiling.
static LogicalResult getConfigForCooperativeMatmul(
    linalg::MatmulOp op, spirv::ResourceLimitsAttr resourceLimits,
    TileSizesListType &tileSizes, std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(op));
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix))
    return failure();

  ShapedType lhsType = op.getOperand(0).getType().cast<ShapedType>();
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ShapedType rhsType = op.getOperand(1).getType().cast<ShapedType>();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  ShapedType outputType = op.getOperand(2).getType().cast<ShapedType>();

  Optional<std::array<int64_t, 3>> coopMatmulSize =
      getCooperativeMatmulSubgroupSize(lhsType.getElementType(),
                                       outputType.getElementType());
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

  // TODO(ravishankarm, antiagainst): For now hardwire the subgroup size.
  const int64_t subgroupSize = 32;
  unsigned maxWorkgroupSize =
      resourceLimits.max_compute_workgroup_invocations().getInt();
  std::tie(numSubgroups[0], numSubgroups[1]) =
      distributeProcs2D(maxWorkgroupSize / subgroupSize);
  numSubgroups[2] = 1;
  // TODO(#3131): This is just being hard-wired for now to be minimal viable,
  // but this can be decided better when we have better estimates of device
  // charecteristics.
  const int64_t numVecMatmulPerSubgroupX = 1;
  const int64_t numVecMatmulPerSubgroupY = 1;
  SmallVector<int64_t, 4> ts = {
      numVecMatmulPerSubgroupY * (*coopMatmulSize)[0] * numSubgroups[1],
      numVecMatmulPerSubgroupX * (*coopMatmulSize)[1] * numSubgroups[0]};
  tileSizes.emplace_back(std::move(ts));

  workgroupSize[0] = numSubgroups[0] * subgroupSize;
  workgroupSize[1] = numSubgroups[1];
  workgroupSize[2] = 1;
  // Subgroup tile sizes
  SmallVector<int64_t, 4> subgroupTs = {
      numVecMatmulPerSubgroupY * (*coopMatmulSize)[0],
      numVecMatmulPerSubgroupX * (*coopMatmulSize)[1], (*coopMatmulSize)[2]};
  tileSizes.emplace_back(std::move(subgroupTs));
  return success();
}

template <>
LogicalResult getOpLaunchConfig(linalg::MatmulOp op,
                                const SPIRVCodegenOptions &options,
                                spirv::ResourceLimitsAttr resourceLimits,
                                TileSizesListType &tileSizes,
                                std::array<int64_t, 3> &workgroupSize,
                                std::array<int64_t, 3> &numSubgroups) {
  if (options.useVectorization &&
      succeeded(getConfigForCooperativeMatmul(op, resourceLimits, tileSizes,
                                              workgroupSize, numSubgroups))) {
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

static LogicalResult getPoolingOpLaunchConfig(
    const SPIRVCodegenOptions &options,
    spirv::ResourceLimitsAttr resourceLimits, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  unsigned maxWorkgroupSize =
      resourceLimits.max_compute_workgroup_invocations().getInt();
  const int64_t tileSizeX = 32;
  int64_t tileSizeY = maxWorkgroupSize / tileSizeX;
  SmallVector<int64_t, 4> ts = {tileSizeY, tileSizeX};
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
    return getPoolingOpLaunchConfig(options, resourceLimits, tileSizes,       \
                                    workgroupSize, numSubgroups);             \
  }

DEFINE_POOLING_OP_CONFIG(linalg::PoolingMaxOp)
DEFINE_POOLING_OP_CONFIG(linalg::PoolingMinOp)
DEFINE_POOLING_OP_CONFIG(linalg::PoolingSumOp)

#undef DEFINE_POOLINGOP_CONFIG

LogicalResult LaunchConfig::init(const SPIRVCodegenOptions &options,
                                 ArrayRef<linalg::LinalgOp> linalgOps) {
  if (!options.workgroupSize.empty()) {
    for (linalg::LinalgOp op : linalgOps)
      tileSizes[op.getOperation()->getName().getStringRef()].emplace_back(
          options.tileSizes.begin(), options.tileSizes.end());
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

  for (linalg::LinalgOp op : linalgOps) {
    StringRef key = op.getOperation()->getName().getStringRef();
    if (tileSizes.count(key)) {
      return op.emitError("unexpected multiple ")
             << key << " operations within dispatch region";
    }

    TileSizesListType &tileSizesInfo = tileSizes[key];

#define DISPATCH(opName)                                                      \
  if (auto lOp = dyn_cast<opName>(op.getOperation())) {                       \
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

}  // namespace iree_compiler
}  // namespace mlir
