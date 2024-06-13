// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree/compiler/Codegen/TransformStrategies/GPU/AbstractGemmLikeStrategy.h"

#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

/// Options to set the default values of the matmul strategy.

static llvm::cl::list<int64_t> clBlockTileSizes(
    "td-matmul-strategy-blk-sizes",
    llvm::cl::desc("block tile size for dims (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::CommaSeparated);
static llvm::cl::opt<int64_t> clReductionTileSize(
    "td-matmul-strategy-reduc-size",
    llvm::cl::desc(
        "reduction tile sized for the transform dialect matmul strategy"));
static llvm::cl::list<int64_t> clNumThreads(
    "td-matmul-strategy-num-threads",
    llvm::cl::desc("number of threads for dims (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::CommaSeparated);
static llvm::cl::list<int64_t> clNumWarps(
    "td-matmul-strategy-num-warps",
    llvm::cl::desc("number of warps for dims (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::CommaSeparated);
static llvm::cl::opt<bool> clUseAsyncCopies(
    "td-matmul-strategy-use-async-copies",
    llvm::cl::desc(
        "use asynchronous copies for the transform dialect matmul strategy"));
static llvm::cl::opt<bool> clUseMmaSync(
    "td-matmul-strategy-use-mma-sync",
    llvm::cl::desc("use mma sync for the transform dialect matmul strategy"));
static llvm::cl::opt<bool> clUseWmma(
    "td-matmul-strategy-use-wmma",
    llvm::cl::desc("use wmma for the transform dialect matmul strategy"));
static llvm::cl::opt<bool> clUseFma(
    "td-matmul-strategy-use-fma",
    llvm::cl::desc("use fma for the transform dialect matmul strategy"));
static llvm::cl::opt<int64_t> clPipelineDepth(
    "td-matmul-strategy-pipeline-depth",
    llvm::cl::desc("pipeline depth for the transform dialect matmul strategy"));
static llvm::cl::opt<bool> clPeelPipelineEpilogue(
    "td-matmul-strategy-peel-pipeline-epilogue",
    llvm::cl::desc("whether to peel the pipeline epilogue for the transform "
                   "dialect matmul strategy"));

using iree_compiler::gpu::AbstractGemmLikeStrategy;

/// Key function for vtable.
AbstractGemmLikeStrategy::~AbstractGemmLikeStrategy() {}

void AbstractGemmLikeStrategy::initDefaultValues(const GPUModel &gpuModel) {
  blockTileSizes =
      SmallVector<int64_t>{clBlockTileSizes.begin(), clBlockTileSizes.end()};
  numThreads = SmallVector<int64_t>{clNumThreads.begin(), clNumThreads.end()};
  numWarps = SmallVector<int64_t>{clNumWarps.begin(), clNumWarps.end()};
  reductionTileSize = clReductionTileSize;
  useAsyncCopies = clUseAsyncCopies;
  useMmaSync = clUseMmaSync;
  useWmma = clUseWmma;
  useFma = clUseFma;
  pipelineDepth = clPipelineDepth;
  peelPipelineEpilogue = clPeelPipelineEpilogue;

  /// cliOptionsSpecified is used to override hard-coded well known good
  /// defaults when set.
  if (clBlockTileSizes.getNumOccurrences() ||
      clNumThreads.getNumOccurrences() || clNumWarps.getNumOccurrences() ||
      clReductionTileSize.getNumOccurrences() ||
      clUseAsyncCopies.getNumOccurrences() ||
      clUseMmaSync.getNumOccurrences() || clUseWmma.getNumOccurrences() ||
      clUseFma.getNumOccurrences() || clPipelineDepth.getNumOccurrences() ||
      clPeelPipelineEpilogue.getNumOccurrences()) {
    cliOptionsSpecified = true;
  }

  /// If not specified, select instructions to target for compute.
  if (!useMmaSync && !useWmma && !useFma) {
    /// First, try to use tensor core.
    if (getLhsElementalType() == getRhsElementalType()) {
      /// Currently all supported targets at least have WMMA.
      /// TODO: Handle targets without tensor core.
      if (gpuModel.hasMmaSync)
        useMmaSync = true;
      else
        useWmma = true;
    } else {
      /// Mixed precision only supported by fma.
      useFma = true;
    }
  }

  /// Prefer smaller subgroup sizes for tensor core strategies.
  if (!useFma)
    targetSubgroupSize = gpuModel.minSubgroupSize;

  /// Default configuration based on hardware properties and problem bit widths.
  if (clBlockTileSizes.getNumOccurrences()) {
    blockTileSizes =
        SmallVector<int64_t>(clBlockTileSizes.begin(), clBlockTileSizes.end());
  } else {
    blockTileSizes = SmallVector<int64_t>{128, 128, 1};
  }

  if (clNumThreads.getNumOccurrences()) {
    numThreads = SmallVector<int64_t>(clNumThreads.begin(), clNumThreads.end());
  } else {
    // Infer from warp counts if present.
    if (clNumWarps.getNumOccurrences()) {
      numThreads = SmallVector<int64_t>(clNumWarps.begin(), clNumWarps.end());
      numThreads[0] *= getSubgroupSize();
    } else {
      numThreads = SmallVector<int64_t>{64, 2, 1};
    }
  }
  if (clNumWarps.getNumOccurrences()) {
    numWarps = SmallVector<int64_t>(clNumWarps.begin(), clNumWarps.end());
  } else {
    numWarps = numThreads;
    numWarps[0] = llvm::divideCeil(numWarps[0], getSubgroupSize());
  }
  if (clUseAsyncCopies.getNumOccurrences())
    useAsyncCopies = clUseAsyncCopies;
  else
    useAsyncCopies = gpuModel.hasMmaSync;
  if (clUseMmaSync.getNumOccurrences())
    useMmaSync = clUseMmaSync;
  if (clUseWmma.getNumOccurrences())
    useWmma = clUseWmma;
  if (clUseFma.getNumOccurrences())
    useFma = clUseFma;
  if (clReductionTileSize.getNumOccurrences()) {
    reductionTileSize = clReductionTileSize;
  } else {
    reductionTileSize = 16;
    if (!useFma) {
      int64_t maxInputWidth =
          std::max(lhsElementalBitWidth(), rhsElementalBitWidth());
      assert(maxInputWidth <= 32 && "requires <= 32-bit types");
      reductionTileSize *= (32 / maxInputWidth);
    }
  }
  if (clPipelineDepth.getNumOccurrences()) {
    pipelineDepth = clPipelineDepth;
  } else {
    pipelineDepth = 0;
    if (useAsyncCopies)
      pipelineDepth = 3;
  }
}

ArrayAttr
AbstractGemmLikeStrategy::getZeroPadAttrFromElementalTypes(OpBuilder &b) const {
  SmallVector<Attribute> paddingValues;
  for (Type t : paddingValueTypes)
    paddingValues.push_back(b.getZeroAttr(t));
  return b.getArrayAttr(paddingValues);
}

//===--------------------------------------------------------------------===//
// Validation of support for the configured strategy.
//===--------------------------------------------------------------------===//

LogicalResult
AbstractGemmLikeStrategy::validate(const GPUModel &gpuModel) const {
  if (totalNumThreads() != totalNumWarps() * getSubgroupSize()) {
    llvm::errs() << "Number of threads specified by warps must match total "
                    "number of threads\n";
    return failure();
  }
  if (m() < blockTileM()) {
    llvm::errs() << "m(" << m() << ") < blockTileM(" << blockTileM() << ") ";
    llvm::errs() << "this is at risk of not vectorizing and is NYI";
    return failure();
  }
  if (n() < blockTileN()) {
    llvm::errs() << "n(" << n() << ") < blockTileN(" << blockTileN() << ") ";
    llvm::errs() << "this is at risk of not vectorizing and is NYI";
    return failure();
  }
  if (k() < reductionTileSize) {
    llvm::errs() << "k(" << k() << ") < reductionTileSize(" << reductionTileSize
                 << ") ";
    llvm::errs() << "this is at risk of not vectorizing and is NYI";
    return failure();
  }

  if (failed(validateLhsCopyMapping())) {
    llvm::errs() << "invalid lhs copy mapping";
    return failure();
  }
  if (failed(validateRhsCopyMapping())) {
    llvm::errs() << "invalid rhs copy mapping";
    return failure();
  }
  if (failed(validateResCopyMapping())) {
    llvm::errs() << "invalid res copy mapping";
    return failure();
  }

  if (pipelineDepth > 1 && reductionTileSize * pipelineDepth > k()) {
    llvm::errs() << "pipeline depth " << pipelineDepth
                 << " too large for reduction tile size " << reductionTileSize
                 << " given k " << k();
    return failure();
  }

  bool oneOption =
      (useMmaSync ^ useWmma ^ useFma) && !(useMmaSync && useWmma && useFma);
  if (!oneOption) {
    llvm::errs() << "at most one of useMmaSync, useWmma, useFma can be true";
    return failure();
  }

  if (useMmaSync) {
    if (blockTileM() < kMinMmaSyncMinM) {
      llvm::errs() << "mma.sync requires at least " << kMinMmaSyncMinM
                   << " block tile size in M";
      return failure();
    }
    if (blockTileN() < kMinMmaSyncMinN) {
      llvm::errs() << "mma.sync requires at least " << kMinMmaSyncMinN
                   << " block tile size in N";
      return failure();
    }
    if (reductionTileSize < kMinMmaSyncMinK) {
      llvm::errs() << "mma.sync requires at least " << kMinMmaSyncMinK
                   << " block tile size in K";
      return failure();
    }
    if (pipelineDepth > 1 && pipelineDepth < kMinMmaSyncPipelineDepth) {
      llvm::errs() << "mma.sync pipelining requires at least "
                   << kMinMmaSyncPipelineDepth << " stages";
      return failure();
    }
    if (pipelineDepth > 1 && reductionTileSize * kMinMmaSyncGroups > k()) {
      llvm::errs() << "mma.sync pipelining requires at least "
                   << kMinMmaSyncGroups << " k groups";
      return failure();
    }
  } else if (useWmma) {
    if (blockTileM() < kMinWmmaMinM) {
      llvm::errs() << "wmma requires at least " << kMinWmmaMinM
                   << " block tile size in M";
      return failure();
    }
    if (blockTileN() < kMinWmmaMinN) {
      llvm::errs() << "wmma requires at least " << kMinWmmaMinN
                   << " block tile size in N";
      return failure();
    }
    if (reductionTileSize < kMinWmmaMinK) {
      llvm::errs() << "wmma requires at least " << kMinWmmaMinK
                   << " block tile size in K";
      return failure();
    }
  }
  return success();
}

//===--------------------------------------------------------------------===//
// Strategy printing for debugging.
//===--------------------------------------------------------------------===//

LLVM_DUMP_METHOD void AbstractGemmLikeStrategy::dump() const {
  print(llvm::errs());
}

void AbstractGemmLikeStrategy::print(llvm::raw_ostream &os) const {
  os << "- forced by CLI specification: "
     << (cliOptionsSpecified ? "true" : "false") << "\n";
  os << "- block tile sizes: {";
  bool isFirst = true;
  for (int64_t blockTileSize : blockTileSizes) {
    if (!isFirst)
      os << ", ";
    os << blockTileSize;
    isFirst = false;
  }
  os << "}\n";
  os << "- reduction tile size: " << reductionTileSize << '\n';

  os << "- number of threads: {";
  isFirst = true;
  for (int64_t numThreadsForDim : numThreads) {
    if (!isFirst)
      os << ", ";
    os << numThreadsForDim;
    isFirst = false;
  }
  os << "}\n";

  os << "- number of warps: {";
  isFirst = true;
  for (int64_t numWarpsForDim : numWarps) {
    if (!isFirst)
      os << ", ";
    os << numWarpsForDim;
    isFirst = false;
  }
  os << "}\n";
  os << "- use async copies: " << useAsyncCopies << '\n';
  os << "- use fma: " << useFma << '\n';
  os << "- use wmma: " << useWmma << '\n';
  os << "- use mma sync: " << useMmaSync << '\n';
  os << "- pipeline depth: " << pipelineDepth << '\n';

  os << "\n-- Derived quantities --\n";
  os << "- lhs copy:\n";
  lhsCopyMapping().print(os << "    -> ");
  os << "\n- rhs copy:\n";
  rhsCopyMapping().print(os << "    -> ");
  os << "\n- res copy:\n";
  resCopyMapping().print(os << "    -> ");
  os << "\n";
}

/// Validates the mapping and emits a diagnostic on failure.
LogicalResult AbstractGemmLikeStrategy::validateCopyMapping(
    MLIRContext *ctx, const MappingInfo &mapping, StringRef name) const {
  int64_t threadsUsed =
      std::accumulate(mapping.numThreads.begin(), mapping.numThreads.end(), 1,
                      std::multiplies<int64_t>());
  if (totalNumThreads() < threadsUsed) {
    InFlightDiagnostic diag = emitError(UnknownLoc::get(ctx))
                              << "too many threads used for transferring "
                              << name;

    std::string str;
    llvm::raw_string_ostream os(str);
    llvm::interleave(mapping.numThreads, os, " * ");
    os << " >= " << totalNumThreads();
    diag.attachNote() << os.str();
    return diag;
  }

  return success();
}
