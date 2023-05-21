// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractGemmLikeStrategy.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

/// Options to set the default values of the matmul strategy.
/// TODO: Replace flag soup in favor of a compilation config attribute similar
/// to the current pipeline based approach.

/// Block tile size X, Y, Z.
static llvm::cl::opt<int64_t> clBlockTileSizeX(
    "td-matmul-strategy-blk-size-x",
    llvm::cl::desc("block tile size for dim X (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(128));
static llvm::cl::opt<int64_t> clBlockTileSizeY(
    "td-matmul-strategy-blk-size-y",
    llvm::cl::desc("block tile size for dim Y (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(128));
static llvm::cl::opt<int64_t> clBlockTileSizeZ(
    "td-matmul-strategy-blk-size-z",
    llvm::cl::desc("block tile size for dim z (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(1));

static llvm::cl::opt<int64_t> clReductionTileSize(
    "td-matmul-strategy-reduc-size",
    llvm::cl::desc(
        "reduction tile sized for the transform dialect matmul strategy"),
    llvm::cl::init(16));

/// Number of threads X, Y, Z.
static llvm::cl::opt<int64_t> clNumThreadsX(
    "td-matmul-strategy-num-threads-x",
    llvm::cl::desc("number of threads for dim X (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(64));
static llvm::cl::opt<int64_t> clNumThreadsY(
    "td-matmul-strategy-num-threads-y",
    llvm::cl::desc("number of threads for dim Y (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(2));
static llvm::cl::opt<int64_t> clNumThreadsZ(
    "td-matmul-strategy-num-threads-z",
    llvm::cl::desc("number of threads for dim z (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(1));

/// Number of warps X, Y, Z.
static llvm::cl::opt<int64_t> clNumWarpsX(
    "td-matmul-strategy-num-warps-x",
    llvm::cl::desc("number of warps for dim X (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(2));
static llvm::cl::opt<int64_t> clNumWarpsY(
    "td-matmul-strategy-num-warps-y",
    llvm::cl::desc("number of warps for dim Y (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(2));
static llvm::cl::opt<int64_t> clNumWarpsZ(
    "td-matmul-strategy-num-warps-z",
    llvm::cl::desc("number of warps for dim z (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(1));

static llvm::cl::opt<bool> clUseAsyncCopies(
    "td-matmul-strategy-use-async-copies",
    llvm::cl::desc(
        "use async copies for the transform dialect matmul strategy"),
    llvm::cl::init(true));

static llvm::cl::opt<int64_t> clPipelineDepth(
    "td-matmul-strategy-pipeline-depth",
    llvm::cl::desc("pipeline depth for the transform dialect matmul strategy"),
    llvm::cl::init(3));

using iree_compiler::gpu::AbstractGemmLikeStrategy;

/// Key function for vtable.
AbstractGemmLikeStrategy::~AbstractGemmLikeStrategy() {}

void AbstractGemmLikeStrategy::initDefaultValues(bool optUseMmaSync) {
  blockTileSizes = {clBlockTileSizeX, clBlockTileSizeY, clBlockTileSizeZ};
  reductionTileSize = clReductionTileSize;
  numThreads = {clNumThreadsX, clNumThreadsY, clNumThreadsZ};
  numWarps = {clNumWarpsX, clNumThreadsY, clNumThreadsZ};
  useAsyncCopies = clUseAsyncCopies;
  useMmaSync = optUseMmaSync;
  pipelineDepth = clPipelineDepth;
}

ArrayAttr AbstractGemmLikeStrategy::getZeroPadAttrFromElementalTypes(
    OpBuilder &b) const {
  SmallVector<Attribute> paddingValues;
  for (Type t : paddingValueTypes) paddingValues.push_back(b.getZeroAttr(t));
  return b.getArrayAttr(paddingValues);
}

/// Prefer 128 bit copies.
constexpr int64_t targetCopyNumBits = 128;

/// Get the largest valid copy vector size up to a vector of size
/// targetCopyNumBits. This assumes the existence of a vector size that can use
/// all of the available threads.
/// TODO: Proper handling of fewer elements than threads.
static int64_t getCopyVectorSize(int64_t innerExtent, int64_t outerCopyExtent,
                                 int64_t innerCopyExtent,
                                 int64_t totalNumThreads, int64_t bitWidth) {
  assert(targetCopyNumBits % bitWidth == 0 &&
         "require bit width that divides target copy num bits");
  int64_t numel = targetCopyNumBits / bitWidth;
  // First adjust for vector alignment on the problem shape.
  while (innerExtent % numel != 0) numel /= 2;

  // Next, adjust for vector alignment on the tile.
  while (innerCopyExtent % numel != 0) numel /= 2;

  // Finally, ensure that the remaining threads will divide the outer dim of the
  // tile.
  int64_t minResidualElements = totalNumThreads / innerCopyExtent;
  assert(outerCopyExtent % minResidualElements == 0 &&
         "fewer elements to copy than total number of threads");
  int64_t maxCopiedElementsPerVector = outerCopyExtent / minResidualElements;
  while (maxCopiedElementsPerVector % numel != 0) numel /= 2;
  return numel;
}

/// Copy vector sizes based on inner most K/N dims.
int64_t AbstractGemmLikeStrategy::lhsCopyVectorSize() const {
  return getCopyVectorSize(k(), blockTileM(), blockTileK(), totalNumThreads(),
                           lhsElementalBitWidth);
}
int64_t AbstractGemmLikeStrategy::rhsCopyVectorSize() const {
  return getCopyVectorSize(n(), blockTileK(), blockTileN(), totalNumThreads(),
                           rhsElementalBitWidth);
}
int64_t AbstractGemmLikeStrategy::resCopyVectorSize() const {
  return getCopyVectorSize(n(), blockTileM(), blockTileN(), totalNumThreads(),
                           resElementalBitWidth);
}

LLVM_DUMP_METHOD void AbstractGemmLikeStrategy::dump() const {
  print(llvm::errs());
}

void AbstractGemmLikeStrategy::print(llvm::raw_ostream &os) const {
  os << "- block tile sizes: {";
  bool isFirst = true;
  for (int64_t blockTileSize : blockTileSizes) {
    if (!isFirst) os << ", ";
    os << blockTileSize;
    isFirst = false;
  }
  os << "}\n";
  os << "- reduction tile size: " << reductionTileSize << '\n';

  os << "- number of threads: {";
  isFirst = true;
  for (int64_t numThreadsForDim : numThreads) {
    if (!isFirst) os << ", ";
    os << numThreadsForDim;
    isFirst = false;
  }
  os << "}\n";

  os << "- number of warps: {";
  isFirst = true;
  for (int64_t numWarpsForDim : numWarps) {
    if (!isFirst) os << ", ";
    os << numWarpsForDim;
    isFirst = false;
  }
  os << "}\n";

  os << "- use async copies: " << useAsyncCopies << '\n';
  os << "- use mma sync: " << useMmaSync << '\n';
  os << "- pipeline depth: " << pipelineDepth << '\n';
}
