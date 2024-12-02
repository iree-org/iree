// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-cpu-types"

namespace mlir::iree_compiler::IREE::CPU {

using Codegen::TileMxNxK;

TileMxNxK chooseMatmulTile(ArrayRef<TileMxNxK> enumeratedTiles,
                           IREE::Encoding::MatmulNarrowDim narrowDim,
                           ArrayRef<int64_t> hostDefinedUpperBound) {
  assert((hostDefinedUpperBound.empty() || hostDefinedUpperBound.size() >= 3) &&
         "expected hostDefinedUpperBound is empty or has upper bound for {M, "
         "N, K}");
  // Handle narrow-N by transposing to reduce to narrow-M. Note: the
  // enumeratedTiles currently only enumerate narrow-M cases.
  if (narrowDim.isN()) {
    SmallVector<int64_t> newHostDefinedUpperBound(hostDefinedUpperBound);
    std::swap(newHostDefinedUpperBound[0], newHostDefinedUpperBound[1]);
    narrowDim.dim = IREE::Encoding::MatmulNarrowDim::Dim::M;
    TileMxNxK tile =
        chooseMatmulTile(enumeratedTiles, narrowDim, newHostDefinedUpperBound);
    std::swap(tile.M, tile.N);
    return tile;
  }
  // Handle kDynamic: currently this is only used with VMVX, where there is only
  // one enumerated tile and it has all three M/N/K dimensions dynamic, so for
  // now we only support that. Generalize that as needed when more dynamic tile
  // sizes are used outside of VMVX, e.g. perhaps some day with Arm SVE. Decide
  // how to incorporate the handling of kDynamic in the cost-model evaluation
  // below to decide when to prefer a dynamic vs a static tile shape.
  for (auto tile : enumeratedTiles) {
    if (ShapedType::isDynamic(tile.M) || ShapedType::isDynamic(tile.N) ||
        ShapedType::isDynamic(tile.K)) {
      assert(enumeratedTiles.size() == 1);
      assert(ShapedType::isDynamic(tile.M) && ShapedType::isDynamic(tile.N) &&
             ShapedType::isDynamic(tile.K));
      return tile;
    }
  }
  // We're going to "rate" the enumerated tiles.
  struct RatedTileMxNxK : TileMxNxK {
    RatedTileMxNxK() {}
    RatedTileMxNxK(TileMxNxK tile) : TileMxNxK(tile) {}
    // Penalize tiles that are wider in the M dimension than matmulNarrowM.
    int64_t paddingPenalty = 0;
    // Favor larger tiles, as long as they still minimize paddingPenalty.
    int64_t productMxNxK = 0;
  };
  SmallVector<RatedTileMxNxK> ratedTiles;
  ratedTiles.reserve(enumeratedTiles.size());
  int64_t bestPaddingPenalty = INT64_MAX;
  int64_t mUB = INT64_MAX;
  int64_t nUB = INT64_MAX;
  int64_t kUB = INT64_MAX;
  if (!hostDefinedUpperBound.empty()) {
    mUB = hostDefinedUpperBound[0];
    nUB = hostDefinedUpperBound[1];
    kUB = hostDefinedUpperBound[2];
  }
  for (auto tile : enumeratedTiles) {
    if (tile.M > mUB || tile.N > nUB || tile.K > kUB) {
      LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "]: tile (";
                 llvm::interleaveComma(
                     ArrayRef<int64_t>{tile.M, tile.N, tile.K}, llvm::dbgs());
                 llvm::dbgs()
                 << ") is skipped because it is not valid for upper_bound (";
                 llvm::interleaveComma(ArrayRef<int64_t>{mUB, nUB, kUB},
                                       llvm::dbgs());
                 llvm::dbgs() << ")\n");
      continue;
    }
    RatedTileMxNxK ratedTile(tile);
    ratedTile.paddingPenalty = 0;
    // If we are choosing a tile for a narrow-M case, we want to minimize
    // padding along the M dimension.
    // The PowerOf2Ceil is so that we are OK with padding up to the next
    // power of two, we just try to avoid padding beyond that. For example,
    // if matmulNarrowM==7 and we have enumerated tiles with M=8,4,2,1, we
    // are OK with the tile that has M==8 even though it requires some padding.
    // Otherwise, we would be penalizing the tiles with M==8,4,2 and we would
    // end up selecting the vecmat tile (M==1) for that case!
    if (narrowDim) {
      ratedTile.paddingPenalty =
          std::max<int64_t>(tile.M - llvm::PowerOf2Ceil(narrowDim.size), 0);
    }
    ratedTile.productMxNxK = tile.M * tile.N * tile.K;
    ratedTiles.push_back(ratedTile);
    LLVM_DEBUG(llvm::dbgs() << "candidate: "; llvm::interleaveComma(
                   ArrayRef<int64_t>{tile.M, tile.N, tile.K}, llvm::dbgs());
               llvm::dbgs() << " penalty:" << ratedTile.paddingPenalty << "\n");
    bestPaddingPenalty = std::min(bestPaddingPenalty, ratedTile.paddingPenalty);
  }
  RatedTileMxNxK bestRatedTile;
  for (auto ratedTile : ratedTiles) {
    // Choose only among tiles that minimize paddingPenalty. Among those,
    // maximize productMxNxK.
    if (ratedTile.paddingPenalty == bestPaddingPenalty &&
        bestRatedTile.productMxNxK < ratedTile.productMxNxK) {
      bestRatedTile = ratedTile;
    }
  }
  // Sanity check. This assert can only fail if there's a programming mistake
  // locally here.
  assert(bestRatedTile.paddingPenalty == bestPaddingPenalty);
  return bestRatedTile;
}

std::optional<StringAttr> getConfigStringAttr(DictionaryAttr config,
                                              StringRef str) {
  if (!config) {
    return std::nullopt;
  }
  auto attr = config.getAs<StringAttr>(str);
  if (!attr) {
    return std::nullopt;
  }
  return attr;
}

std::optional<llvm::Triple> getTargetTriple(DictionaryAttr config) {
  auto triple = getConfigStringAttr(config, "target_triple");
  if (!triple) {
    return std::nullopt;
  }
  return llvm::Triple(triple.value().str());
}

static std::optional<StringRef> getCpuFeatures(DictionaryAttr config) {
  auto cpuFeatures = getConfigStringAttr(config, "cpu_features");
  if (!cpuFeatures) {
    return std::nullopt;
  }
  return cpuFeatures->getValue();
}

// TODO: If we have to check for a significantly large number of features in the
// future, we may want to consider a persistent state to carry over processed
// HAL information or keeping the TTI instance alive and query subtarget
// features data structure.
bool hasFeature(DictionaryAttr config, StringRef feature) {
  std::optional<StringRef> features = getCpuFeatures(config);
  if (!features) {
    return false;
  }

  // Find feature string in list of features, making sure that we don't match a
  // sub-string.
  std::stringstream sstream(features->str());
  std::string str;
  while (std::getline(sstream, str, ',')) {
    if (str == feature) {
      return true;
    }
  }

  return false;
}

bool isX86(DictionaryAttr config) {
  std::optional<llvm::Triple> triple = getTargetTriple(config);
  return triple && triple.value().isX86();
}

bool isX86_64(DictionaryAttr config) {
  std::optional<llvm::Triple> triple = getTargetTriple(config);
  return triple && triple.value().getArch() == llvm::Triple::x86_64;
}

bool isAArch64(DictionaryAttr config) {
  std::optional<llvm::Triple> triple = getTargetTriple(config);
  return triple && triple.value().isAArch64();
}

bool isRISCV(DictionaryAttr config) {
  std::optional<llvm::Triple> triple = getTargetTriple(config);
  return triple && triple.value().isRISCV();
}

bool isRISCV32(DictionaryAttr config) {
  std::optional<llvm::Triple> triple = getTargetTriple(config);
  return triple && triple.value().isRISCV32();
}

} // namespace mlir::iree_compiler::IREE::CPU
