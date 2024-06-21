// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/GPU/CopyMapping.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "iree-gpu-copy-mapping"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

int64_t iree_compiler::gpu::CopyMapping::maxContiguousElementsToTransfer(
    int64_t alignment, int64_t numContiguousElements,
    int64_t elementalBitWidth) {
  assert(kCudaMaxVectorLoadBitWidth % elementalBitWidth == 0 &&
         "elemental bitwidth does not divide kCudaMaxVectorLoadBitWidth");
  return std::gcd(std::gcd(alignment, numContiguousElements),
                  kCudaMaxVectorLoadBitWidth / elementalBitWidth);
}

FailureOr<iree_compiler::gpu::CopyMapping>
iree_compiler::gpu::CopyMapping::numThreadsForCopy(int totalNumThreads,
                                                   int64_t alignment,
                                                   ArrayRef<int64_t> sizes,
                                                   bool favorPredication,
                                                   int64_t elementalBitWidth) {
  LDBG("\nSTART numThreadsForCopy, favorPredication: " << favorPredication);
  LLVM_DEBUG(llvm::interleaveComma(sizes, DBGS() << "--sizes: ");
             llvm::dbgs() << "\n";);

  // Greedily find the largest vector size that can be used to copy the most
  // minor dimension: we are in the business of filling 128B contiguous memory
  // transactions with as few threads as possible.
  int64_t maxVectorSize = CopyMapping::maxContiguousElementsToTransfer(
      alignment, sizes.back(), elementalBitWidth);
  LDBG("--maxVectorSize: " << maxVectorSize);
  int64_t numElements = 1;
  for (auto s : sizes)
    numElements *= s;
  LDBG("--numElements: " << numElements);

  int64_t actualVectorSize = maxVectorSize;
  if (!favorPredication) {
    // Bias towards reducing the vector size to avoid predication.
    // Predication occurs if we end up using fewer than totalNumThreads for a
    // particular copy.
    // Predication chokes the current implementation of shared memory
    // pipelining.
    // TODO: Reevaluate this heuristic when we have a more robust pipelining
    // implementation.
    for (; actualVectorSize >= 1; actualVectorSize /= 2) {
      LDBG("--step totalNumThreads * actualVectorSize: "
           << totalNumThreads * actualVectorSize);
      if (numElements % (totalNumThreads * actualVectorSize) != 0)
        continue;
      break;
    }
    LDBG("--numElements: " << numElements);
    LDBG("--totalNumThreads: " << totalNumThreads);
    LDBG("--actualVectorSize: " << actualVectorSize);
    if (actualVectorSize == 0) {
      LDBG("--Could not map copy without predication -> FAIL");
      return failure();
    }
  }

  // Scale back the last size by actualVectorSize to account for the fact
  // that we perform vector transfers.
  assert(sizes.back() % actualVectorSize == 0 &&
         "most-minor size not divisible by actualVectorSize");
  SmallVector<int64_t> scaledSizes{sizes.begin(), sizes.end()};
  scaledSizes.back() /= actualVectorSize;

  int64_t numThreadsRemaining = totalNumThreads;
  LDBG("--numThreadsRemaining: " << numThreadsRemaining);
  SmallVector<int64_t> factors;
  for (auto s : llvm::reverse(scaledSizes)) {
    int64_t gcd = std::gcd(numThreadsRemaining, s);
    factors.push_back(gcd);
    numThreadsRemaining /= gcd;
    LDBG("--new factors: " << gcd);
    LDBG("--numThreadsRemaining: " << numThreadsRemaining);
  }

  std::reverse(factors.begin(), factors.end());

  LLVM_DEBUG(llvm::interleaveComma(factors, DBGS() << "numThreads: ");
             llvm::dbgs() << "\n";
             LDBG("actualVectorSize: " << actualVectorSize););

  return CopyMapping{actualVectorSize, factors};
}

iree_compiler::gpu::MappingInfo iree_compiler::gpu::CopyMapping::getMappingInfo(
    MLIRContext *ctx, int totalNumThreads, int64_t alignment,
    ArrayRef<int64_t> copySizes, bool favorPredication,
    int64_t elementalBitWidth) {
  assert(!copySizes.empty() && copySizes.size() <= 3 &&
         "only 1,2,3-D copies are supported for now");
  FailureOr<CopyMapping> maybeCopyMapping =
      CopyMapping::numThreadsForCopy(totalNumThreads, alignment, copySizes,
                                     favorPredication, elementalBitWidth);
  // If failed, try again with predication; this must succeed.
  if (failed(maybeCopyMapping)) {
    assert(!favorPredication &&
           "maybe copy mapping may not fail with predication");
    maybeCopyMapping = CopyMapping::numThreadsForCopy(
        totalNumThreads, alignment, copySizes, /*favorPredication=*/true,
        elementalBitWidth);
  }
  assert(succeeded(maybeCopyMapping) && "failed to compute copy mapping");
  assert(maybeCopyMapping->numThreads.size() == copySizes.size() &&
         "compute copy mapping expected same number of threads and copy sizes");

  SmallVector<int64_t> tileSizes = llvm::to_vector(llvm::map_range(
      llvm::zip(copySizes, maybeCopyMapping->numThreads), [](auto &&pair) {
        int64_t size, numThreads;
        std::tie(size, numThreads) = pair;
        return llvm::divideCeilSigned(size, numThreads);
      }));
  SmallVector<Attribute> allThreadMappings{linearId2(ctx), linearId1(ctx),
                                           linearId0(ctx)};
  auto threadMapping =
      llvm::to_vector(ArrayRef(allThreadMappings).take_back(tileSizes.size()));

  MappingInfo res{/*numThreads=*/maybeCopyMapping->numThreads,
                  /*tilecopySizes=*/tileSizes,
                  /*threadMapping=*/threadMapping,
                  /*vectorSize=*/maybeCopyMapping->vectorSize};
  LLVM_DEBUG(res.print(DBGS()); llvm::dbgs() << "\n");
  return res;
}
