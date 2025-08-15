// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"

#include <optional>
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE::GPU {

namespace {

//===----------------------------------------------------------------------===//
// Target details structs
//===----------------------------------------------------------------------===//

// Note that the following structs basically mirror the corresponding attribute
// definitions in IREEGPUAttrs.td. The purpose of having mirroring structs here
// is to allow us building a known GPU target "database" in POD data structures
// without all the MLIR attribute wrappers; so life is easier to put such
// information in static constant data.
//
// Ideally we should be able to generate The following structs from the TableGen
// attribute definitions, but it takes quite some efforts to plumb all pieces
// through. So now fine with some duplication.

// Workgroup processor level feature/limit details
struct WgpDetails {
  ComputeBitwidths compute;
  StorageBitwidths storage;
  SubgroupOps subgroupOps;
  DotProductOps dotproductOps;
  uint32_t mmaCount;
  const MMAIntrinsic *mmaOps;
  uint32_t scaledMmaCount;
  const ScaledMMAIntrinsic *scaledMmaOps;

  // We support two static values here mostly due to AMD RDNA GPUs have two
  // modes. Use duplicated values if the GPU only have one subgroup size.
  std::array<int32_t, 2> subgroupSizeChoices;
  std::array<int32_t, 3> maxWorkgroupSizes;
  int32_t maxThreadSize;
  int32_t maxWorkgroupMemoryBytes;
  std::array<int32_t, 3> maxWorkgroupCounts;
  std::optional<int32_t> maxLoadInstructionBits;
  std::optional<int32_t> simdsPerWgp;
  std::optional<int32_t> vgprSpaceBits;
};

// Chip level feature/limit details
struct ChipDetails {
  uint32_t wgpCount;
  std::optional<StringRef> sku;
  // Aggregate chip-level bandwidth in TB/s.
  std::optional<float> peakMemoryBandwidthTBs;
  // Optional per-data-type compute performance (TFLOPs/s).
  llvm::SmallDenseMap<ComputeBitwidths, float> peakPerfTFLOPs;

  ChipDetails(
      uint32_t wgp, std::optional<llvm::StringRef> s = std::nullopt,
      std::optional<float> bw = std::nullopt,
      std::initializer_list<llvm::detail::DenseMapPair<ComputeBitwidths, float>>
          perf = {})
      : wgpCount(wgp), sku(s), peakMemoryBandwidthTBs(bw),
        peakPerfTFLOPs(perf) {}
};

// Full target details
struct TargetDetails {
  const WgpDetails *wgp;
  const ChipDetails *chip;
};

//===----------------------------------------------------------------------===//
// Utility definitions
//===----------------------------------------------------------------------===//

const ComputeBitwidths allFPComputeBits =
    ComputeBitwidths::FP64 | ComputeBitwidths::FP32 | ComputeBitwidths::FP16;
const ComputeBitwidths allIntComputeBits =
    ComputeBitwidths::Int64 | ComputeBitwidths::Int32 |
    ComputeBitwidths::Int16 | ComputeBitwidths::Int8;
const ComputeBitwidths allComputeBits = allFPComputeBits | allIntComputeBits;

const StorageBitwidths allStorageBits =
    StorageBitwidths::B8 | StorageBitwidths::B16 | StorageBitwidths::B32 |
    StorageBitwidths::B64;

const SubgroupOps allSubgroupOps =
    SubgroupOps::Shuffle | SubgroupOps::Arithmetic;

const DotProductOps allDotProductOps = DotProductOps::DP4xI8ToI32;

#define ARRAY_SIZE(array) (sizeof(array) / sizeof(array[0]))

// Creates the corresponding TargetAttr from the given target |details|.
TargetAttr createTargetAttr(const TargetDetails &details, StringRef arch,
                            StringRef features, MLIRContext *context) {
  const WgpDetails *wgp = details.wgp;

  SmallVector<MMAAttr, 8> mmaAttrs;
  mmaAttrs.reserve(wgp->mmaCount);
  for (int i = 0; i < wgp->mmaCount; ++i)
    mmaAttrs.push_back(MMAAttr::get(context, wgp->mmaOps[i]));

  SmallVector<ScaledMMAAttr, 8> scaledMmaAttrs;
  scaledMmaAttrs.reserve(wgp->scaledMmaCount);
  for (int i = 0; i < wgp->scaledMmaCount; ++i) {
    for (auto inTy : ScaledMMAAttr::getSupportedInputTypes(context)) {
      for (auto outTy : ScaledMMAAttr::getSupportedOutputTypes(context)) {
        scaledMmaAttrs.push_back(
            ScaledMMAAttr::get(context, wgp->scaledMmaOps[i], inTy, inTy, outTy,
                               /*columnMajor=*/false));
      }
    }
  }
  SmallVector<int32_t, 2> subgroupSizes;
  assert(wgp->subgroupSizeChoices.front() != 0);
  assert(wgp->subgroupSizeChoices.back() != 0);
  subgroupSizes.push_back(wgp->subgroupSizeChoices.front());
  if (wgp->subgroupSizeChoices.back() != wgp->subgroupSizeChoices.front()) {
    subgroupSizes.push_back(wgp->subgroupSizeChoices.back());
  }

  auto targetWgp = TargetWgpAttr::get(
      context, ComputeBitwidthsAttr::get(context, details.wgp->compute),
      StorageBitwidthsAttr::get(context, wgp->storage),
      SubgroupOpsAttr::get(context, wgp->subgroupOps),
      DotProductOpsAttr::get(context, wgp->dotproductOps),
      MMAOpsArrayAttr::get(context, mmaAttrs),
      ScaledMMAOpsArrayAttr::get(context, scaledMmaAttrs),
      DenseI32ArrayAttr::get(context, subgroupSizes),
      DenseI32ArrayAttr::get(context, wgp->maxWorkgroupSizes),
      wgp->maxThreadSize, wgp->maxWorkgroupMemoryBytes,
      DenseI32ArrayAttr::get(context, wgp->maxWorkgroupCounts),
      wgp->maxLoadInstructionBits, wgp->simdsPerWgp, wgp->vgprSpaceBits,
      DictionaryAttr{});

  TargetChipAttr targetChip;
  if (details.chip) {
    auto skuAttr = details.chip->sku
                       ? StringAttr::get(context, *details.chip->sku)
                       : StringAttr{};

    FloatAttr peakMemoryBandwidthAttr =
        details.chip->peakMemoryBandwidthTBs
            ? FloatAttr::get(Float32Type::get(context),
                             *details.chip->peakMemoryBandwidthTBs)
            : FloatAttr{};

    DictionaryAttr peakPerfTFLOPsAttr = {};
    if (!details.chip->peakPerfTFLOPs.empty()) {
      SmallVector<NamedAttribute> attributes = llvm::map_to_vector(
          details.chip->peakPerfTFLOPs, [&](const auto &pair) {
            return NamedAttribute(
                stringifyComputeBitwidths(pair.first),
                FloatAttr::get(Float32Type::get(context), pair.second));
          });
      peakPerfTFLOPsAttr = DictionaryAttr::get(context, attributes);
    }
    targetChip = TargetChipAttr::get(context, details.chip->wgpCount, skuAttr,
                                     peakMemoryBandwidthAttr,
                                     peakPerfTFLOPsAttr, DictionaryAttr{});
  }

  return TargetAttr::get(context, arch, features, targetWgp, targetChip);
}

//===----------------------------------------------------------------------===//
// Known AMD target details
//
// Note: the max workgroup size is given as signed int32 max because MLIR's
// `index` is signed and the workgroup ID is sign-extended, not zero-extended,
// to 64-bits.
//===----------------------------------------------------------------------===//

const WgpDetails *getCDNA4WgpDetails() {
  static const MMAIntrinsic cdna4MMAOps[] = {
      // Introduced in CDNA4
      MMAIntrinsic::MFMA_F32_16x16x32_F16,
      MMAIntrinsic::MFMA_F32_32x32x16_F16,
      MMAIntrinsic::MFMA_F32_16x16x32_BF16,
      MMAIntrinsic::MFMA_F32_32x32x16_BF16,
      MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2,
      MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2_F8E4M3FN,
      MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN,
      MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN_F8E5M2,
      MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2,
      MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2_F8E4M3FN,
      MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN,
      MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN_F8E5M2,
      MMAIntrinsic::MFMA_I32_16x16x64_I8,
      MMAIntrinsic::MFMA_I32_32x32x32_I8,
      // Introduced in CDNA3
      MMAIntrinsic::MFMA_F32_16x16x16_BF16,
      MMAIntrinsic::MFMA_F32_32x32x8_BF16,
      // Note: use same instructions as in CDNA3 but different types
      MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2,
      MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2_F8E4M3FN,
      MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FN,
      MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FN_F8E5M2,
      MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2,
      MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2_F8E4M3FN,
      MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FN,
      MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FN_F8E5M2,
      MMAIntrinsic::MFMA_I32_16x16x32_I8,
      MMAIntrinsic::MFMA_I32_32x32x16_I8,
      // Introduced in CDNA2, still present in CDNA3
      MMAIntrinsic::MFMA_F64_16x16x4_F64,
      // Introduced in CDNA1, still present in CDNA3
      MMAIntrinsic::MFMA_F32_16x16x4_F32,
      MMAIntrinsic::MFMA_F32_16x16x16_F16,
      MMAIntrinsic::MFMA_F32_32x32x8_F16,
  };
  static const ScaledMMAIntrinsic cdna4ScaledMMAOps[] = {
      // Introduced in CDNA4
      ScaledMMAIntrinsic::MFMA_SCALE_F32_16x16x128_B32,
      ScaledMMAIntrinsic::MFMA_SCALE_F32_32x32x64_B32,
  };
  static const WgpDetails cdna4Wgp = {allComputeBits,
                                      allStorageBits,
                                      allSubgroupOps,
                                      allDotProductOps,
                                      ARRAY_SIZE(cdna4MMAOps),
                                      cdna4MMAOps,
                                      ARRAY_SIZE(cdna4ScaledMMAOps),
                                      cdna4ScaledMMAOps,
                                      {64, 64},
                                      {1024, 1024, 1024},
                                      1024,
                                      // Note: upgraded from CDNA3
                                      160 * 1024,
                                      {0x7fffffff, 0x7fffffff, 0x7fffffff},
                                      /*maxLoadInstructionBits=*/128,
                                      /*simdsPerWgp=*/4,
                                      /*vgprSpaceBits=*/512 * 32};
  return &cdna4Wgp;
}

const WgpDetails *getCDNA3WgpDetails() {
  // Note: these operations are listed in order of preference.
  static const MMAIntrinsic cdna3MMAOps[] = {
      // Introduced in CDNA3
      MMAIntrinsic::MFMA_F32_16x16x16_BF16,
      MMAIntrinsic::MFMA_F32_32x32x8_BF16,
      MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ,
      MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ,
      MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ,
      MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ,
      MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2FNUZ,
      MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ,
      MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ,
      MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ,
      MMAIntrinsic::MFMA_I32_16x16x32_I8,
      MMAIntrinsic::MFMA_I32_32x32x16_I8,
      // Introduced in CDNA2, still present in CDNA3
      MMAIntrinsic::MFMA_F64_16x16x4_F64,
      // Introduced in CDNA1, still present in CDNA3
      MMAIntrinsic::MFMA_F32_16x16x4_F32,
      MMAIntrinsic::MFMA_F32_16x16x16_F16,
      MMAIntrinsic::MFMA_F32_32x32x8_F16,
  };
  static const WgpDetails cdna3Wgp = {allComputeBits,
                                      allStorageBits,
                                      allSubgroupOps,
                                      allDotProductOps,
                                      ARRAY_SIZE(cdna3MMAOps),
                                      cdna3MMAOps,
                                      0,
                                      nullptr,
                                      {64, 64},
                                      {1024, 1024, 1024},
                                      1024,
                                      64 * 1024,
                                      {0x7fffffff, 0x7fffffff, 0x7fffffff},
                                      /*maxLoadInstructionBits=*/128,
                                      /*simdsPerWgp=*/4,
                                      /*vgprSpaceBits=*/512 * 32};
  return &cdna3Wgp;
}

const WgpDetails *getCDNA2WgpDetails() {
  static const MMAIntrinsic cdna2MMAOps[] = {
      // Introduced in CDNA2
      MMAIntrinsic::MFMA_F32_16x16x8_BF16,
      MMAIntrinsic::MFMA_F32_32x32x4_BF16,
      MMAIntrinsic::MFMA_F64_16x16x4_F64,
      // Introduced in CDNA1
      MMAIntrinsic::MFMA_F32_16x16x4_F32,
      MMAIntrinsic::MFMA_F32_16x16x16_F16,
      MMAIntrinsic::MFMA_F32_32x32x8_F16,
      MMAIntrinsic::MFMA_I32_16x16x16_I8,
      MMAIntrinsic::MFMA_I32_32x32x8_I8,
  };
  static const WgpDetails cdna2Wgp = {allComputeBits,
                                      allStorageBits,
                                      allSubgroupOps,
                                      allDotProductOps,
                                      ARRAY_SIZE(cdna2MMAOps),
                                      cdna2MMAOps,
                                      0,
                                      nullptr,
                                      {64, 64},
                                      {1024, 1024, 1024},
                                      1024,
                                      64 * 1024,
                                      {0x7fffffff, 0x7fffffff, 0x7fffffff},
                                      /*maxLoadInstructionBits=*/128,
                                      /*simdsPerWgp=*/4,
                                      /*vgprSpaceBits=*/256 * 32};
  return &cdna2Wgp;
}

const WgpDetails *getCDNA1WgpDetails() {
  static const MMAIntrinsic cdna1MMAOps[] = {
      MMAIntrinsic::MFMA_F32_16x16x4_F32, MMAIntrinsic::MFMA_F32_16x16x16_F16,
      MMAIntrinsic::MFMA_F32_32x32x8_F16, MMAIntrinsic::MFMA_I32_16x16x16_I8,
      MMAIntrinsic::MFMA_I32_32x32x8_I8,
  };
  static const WgpDetails cdna1Wgp = {allComputeBits,
                                      allStorageBits,
                                      allSubgroupOps,
                                      allDotProductOps,
                                      ARRAY_SIZE(cdna1MMAOps),
                                      cdna1MMAOps,
                                      0,
                                      nullptr,
                                      {64, 64},
                                      {1024, 1024, 1024},
                                      1024,
                                      64 * 1024,
                                      {0x7fffffff, 0x7fffffff, 0x7fffffff},
                                      /*maxLoadInstructionBits=*/128,
                                      /*simdsPerWgp=*/4,
                                      /*vgprSpaceBits=*/256 * 32};
  return &cdna1Wgp;
}

const WgpDetails *getRDNA4WgpDetails() {
  static const MMAIntrinsic rdna4MMAOps[] = {
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      MMAIntrinsic::WMMAR4_F16_16x16x16_F16,
      MMAIntrinsic::WMMAR4_F32_16x16x16_BF16,
      MMAIntrinsic::WMMAR4_BF16_16x16x16_BF16,
      MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2,
      MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN,
      MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN,
      MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2,
      MMAIntrinsic::WMMAR4_I32_16x16x16_I8,

  };
  static const WgpDetails rdna4Wgp = {allComputeBits,
                                      allStorageBits,
                                      allSubgroupOps,
                                      allDotProductOps,
                                      ARRAY_SIZE(rdna4MMAOps),
                                      rdna4MMAOps,
                                      0,
                                      nullptr,
                                      {32, 64},
                                      {1024, 1024, 1024},
                                      1024,
                                      64 * 1024,
                                      {0x7fffffff, 0x7fffffff, 0x7fffffff},
                                      /*maxLoadInstructionBits=*/128,
                                      /*simdsPerWgp=*/4,
                                      /*vgprSpaceBits=*/256 * 32};
  return &rdna4Wgp;
}

const WgpDetails *getRDNA3WgpDetails() {
  static const MMAIntrinsic rdna3MMAOps[] = {
      MMAIntrinsic::WMMAR3_F32_16x16x16_F16,
      MMAIntrinsic::WMMAR3_F16_16x16x16_F16,
      MMAIntrinsic::WMMAR3_F32_16x16x16_BF16,
      MMAIntrinsic::WMMAR3_BF16_16x16x16_BF16,
      MMAIntrinsic::WMMAR3_I32_16x16x16_I8,

  };
  static const WgpDetails rdna3Wgp = {allComputeBits,
                                      allStorageBits,
                                      allSubgroupOps,
                                      allDotProductOps,
                                      ARRAY_SIZE(rdna3MMAOps),
                                      rdna3MMAOps,
                                      0,
                                      nullptr,
                                      {32, 64},
                                      {1024, 1024, 1024},
                                      1024,
                                      64 * 1024,
                                      {0x7fffffff, 0x7fffffff, 0x7fffffff},
                                      /*maxLoadInstructionBits=*/128,
                                      /*simdsPerWgp=*/4,
                                      /*vgprSpaceBits=*/256 * 32};
  return &rdna3Wgp;
}

const WgpDetails *getRDNA2WgpDetails() {
  static const WgpDetails rdna2Wgp = {allComputeBits,
                                      allStorageBits,
                                      allSubgroupOps,
                                      allDotProductOps,
                                      /*mmaCount=*/0,
                                      /*mmaOps=*/nullptr,
                                      /*scaledMmaCount=*/0,
                                      /*scaledMmaOps=*/nullptr,
                                      {32, 64},
                                      {1024, 1024, 1024},
                                      1024,
                                      64 * 1024,
                                      {0x7fffffff, 0x7fffffff, 0x7fffffff}};
  return &rdna2Wgp;
}

const WgpDetails *getRDNA1WgpDetails() {
  static const WgpDetails rdna1Wgp = {allComputeBits,
                                      allStorageBits,
                                      allSubgroupOps,
                                      DotProductOps::None,
                                      /*mmaCount=*/0,
                                      /*mmaOps=*/nullptr,
                                      /*scaledMmaCount=*/0,
                                      /*scaledMmaOps=*/nullptr,
                                      {32, 64},
                                      {1024, 1024, 1024},
                                      1024,
                                      64 * 1024,
                                      {0x7fffffff, 0x7fffffff, 0x7fffffff}};
  return &rdna1Wgp;
}

std::optional<TargetDetails> getAMDGPUTargetDetails(StringRef target) {
  const WgpDetails *cdna4Wgp = getCDNA4WgpDetails();
  const WgpDetails *cdna3Wgp = getCDNA3WgpDetails();
  const WgpDetails *cdna2Wgp = getCDNA2WgpDetails();
  const WgpDetails *cdna1Wgp = getCDNA1WgpDetails();
  const WgpDetails *rdna4Wgp = getRDNA4WgpDetails();
  const WgpDetails *rdna3Wgp = getRDNA3WgpDetails();
  const WgpDetails *rdna2Wgp = getRDNA2WgpDetails();
  const WgpDetails *rdna1Wgp = getRDNA1WgpDetails();

  // --- CDNA --- //
  // "AMD Instinct MI350 Series Product Offerings" in Page 18 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
  static const ChipDetails mi350xChip = {256,
                                         "mi350x",
                                         8.0f,
                                         {{ComputeBitwidths::FP32, 144.2f},
                                          {ComputeBitwidths::FP16, 2300.0f},
                                          {ComputeBitwidths::Int8, 4600.0f},
                                          {ComputeBitwidths::FP8, 4600.0f},
                                          {ComputeBitwidths::FP6, 9200.0f},
                                          {ComputeBitwidths::FP4, 9200.0f}}};

  static const ChipDetails mi355xChip = {256,
                                         "mi355x",
                                         8.0f,
                                         {{ComputeBitwidths::FP32, 157.3f},
                                          {ComputeBitwidths::FP16, 2500.0f},
                                          {ComputeBitwidths::Int8, 5000.0f},
                                          {ComputeBitwidths::FP8, 5000.0f},
                                          {ComputeBitwidths::FP6, 10000.0f},
                                          {ComputeBitwidths::FP4, 10000.0f}}};

  // "AMD Instinct MI300 Series Product Offerings" in Page 23 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
  static const ChipDetails mi300xChip = {304,
                                         "mi300x",
                                         5.3f,
                                         {{ComputeBitwidths::FP32, 163.4f},
                                          {ComputeBitwidths::FP16, 1307.4f},
                                          {ComputeBitwidths::Int8, 2614.9f},
                                          {ComputeBitwidths::FP8, 2614.9f}}};

  static const ChipDetails mi300aChip = {228,
                                         "mi300a",
                                         5.3f,
                                         {{ComputeBitwidths::FP32, 122.6f},
                                          {ComputeBitwidths::FP16, 980.6f},
                                          {ComputeBitwidths::Int8, 1961.2f},
                                          {ComputeBitwidths::FP8, 1961.2f}}};

  static const ChipDetails mi308xChip = {
      80,
      "mi308x",
      5.3f,
      // Peak fp32 perf estimated from:
      // 80(CUs)*4(SIMDs)*1.42(Freq)*(16*16*4)(GEMM shape)*2(mul+add)/32(latency
      // instruction)
      {{ComputeBitwidths::FP32, 29.0f},
       {ComputeBitwidths::FP16, 188.4f},
       {ComputeBitwidths::FP8, 176.8f},
       // Estimated int8 performance based on FP8
       {ComputeBitwidths::Int8, 176.8f}}};

  static const ChipDetails mi325xChip = {304,
                                         "mi325x",
                                         5.3f,
                                         {{ComputeBitwidths::FP32, 163.4f},
                                          {ComputeBitwidths::FP16, 1307.4f},
                                          {ComputeBitwidths::Int8, 2614.9f},
                                          {ComputeBitwidths::FP8, 2614.9f}}};

  // "AMD Instinct MI200 Series Accelerator Product Offerings" in Page 14 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
  static const ChipDetails mi250xChip = {220,
                                         "mi250x",
                                         3.2f,
                                         {{ComputeBitwidths::FP32, 95.7f},
                                          {ComputeBitwidths::FP16, 383.0f},
                                          {ComputeBitwidths::Int8, 383.0f}}};

  static const ChipDetails mi250Chip = {208,
                                        "mi250",
                                        3.2f,
                                        {{ComputeBitwidths::FP32, 90.5f},
                                         {ComputeBitwidths::FP16, 362.1f},
                                         {ComputeBitwidths::Int8, 362.1f}}};
  static const ChipDetails mi210Chip = {104,
                                        "mi210",
                                        1.6f,
                                        {{ComputeBitwidths::FP32, 45.3f},
                                         {ComputeBitwidths::FP16, 181.0f},
                                         {ComputeBitwidths::Int8, 181.0f}}};

  // "AMD CDNA Architecture Compute Units" in Page 5 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
  static const ChipDetails mi100Chip = {120,
                                        "mi100",
                                        1.23f,
                                        {{ComputeBitwidths::FP32, 46.1f},
                                         {ComputeBitwidths::FP16, 184.6f},
                                         {ComputeBitwidths::Int8, 184.6f}}};

  // --- RDNA --- //

  // With RDNA, two Compute Units form a Workgroup Processor (WGP).
  // A kernel can be dispatched in either the CU mode or the WGP mode (where
  // some resources like LDS are accessible to both CUs). The default with HIP
  // is the WGP mode. For the purpose of distribution heuristics, we divide the
  // number of CU reported in the hardware spaces by two to get the number of
  // WGPs.

  // AMD RDNA4 architecture:
  // https://www.amd.com/en/newsroom/press-releases/2025-2-28-amd-unveils-next-generation-amd-rdna-4-architectu.html.
  // https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html
  static const ChipDetails r9700Chip = {64 / 2,
                                        "r9700",
                                        0.64f,
                                        {{ComputeBitwidths::FP32, 47.8f},
                                         {ComputeBitwidths::FP16, 191.0f},
                                         {ComputeBitwidths::Int8, 383.0f},
                                         {ComputeBitwidths::FP8, 383.0f}}};
  // https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html
  static const ChipDetails rx9070xtChip = {64 / 2,
                                           "rx9070xt",
                                           0.64f,
                                           {{ComputeBitwidths::FP32, 48.7f},
                                            {ComputeBitwidths::FP16, 195.0f},
                                            {ComputeBitwidths::Int8, 389.0f},
                                            {ComputeBitwidths::FP8, 389.0f}}};
  // https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html
  static const ChipDetails rx9070Chip = {56 / 2,
                                         "rx9070",
                                         0.64f,
                                         {{ComputeBitwidths::FP32, 36.1f},
                                          {ComputeBitwidths::FP16, 145.0f},
                                          {ComputeBitwidths::Int8, 289.0f},
                                          {ComputeBitwidths::FP8, 289.0f}}};
  // https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html
  static const ChipDetails rx9060xtChip = {32 / 2,
                                           "rx9060xt",
                                           0.32f,
                                           {{ComputeBitwidths::FP32, 25.6f},
                                            {ComputeBitwidths::FP16, 103.0f},
                                            {ComputeBitwidths::Int8, 205.0f},
                                            {ComputeBitwidths::FP8, 205.0f}}};

  // AMD RDNA3.
  static const ChipDetails rx7900xtxChip = {96 / 2, "rx7900xtx"};
  static const ChipDetails rx7900xtChip = {84 / 2, "rx7900xt"};
  static const ChipDetails rx7800xtChip = {60 / 2, "rx7800xt"};
  static const ChipDetails rx7700xtChip = {54 / 2, "rx7700xt"};
  static const ChipDetails v710Chip = {54 / 2, "v710"};
  static const ChipDetails w7900Chip = {96 / 2, "w7900"};
  static const ChipDetails w7800Chip = {70 / 2, "w7800"};
  static const ChipDetails w7700Chip = {48 / 2, "w7700"};

  // See https://llvm.org/docs/AMDGPUUsage.html#processors for gfxN to
  // cdnaN/rdnaN mapping.
  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      .Case("mi355x", TargetDetails{cdna4Wgp, &mi355xChip})
      .Case("mi350x", TargetDetails{cdna4Wgp, &mi350xChip})
      .Cases("cdna4", "gfx950", TargetDetails{cdna4Wgp, nullptr})
      .Case("mi325x", TargetDetails{cdna3Wgp, &mi325xChip})
      .Case("mi300x", TargetDetails{cdna3Wgp, &mi300xChip})
      .Case("mi300a", TargetDetails{cdna3Wgp, &mi300aChip})
      .Case("mi308x", TargetDetails{cdna3Wgp, &mi308xChip})
      .Cases("cdna3", "gfx942", TargetDetails{cdna3Wgp, nullptr})
      .Case("mi250x", TargetDetails{cdna2Wgp, &mi250xChip})
      .Case("mi250", TargetDetails{cdna2Wgp, &mi250Chip})
      .Case("mi210", TargetDetails{cdna2Wgp, &mi210Chip})
      .Cases("cdna2", "gfx90a", TargetDetails{cdna2Wgp, nullptr})
      .Case("mi100", TargetDetails{cdna1Wgp, &mi100Chip})
      .Cases("cdna1", "gfx908", TargetDetails{cdna1Wgp, nullptr})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-9070-xt.c4229
      .Case("rx9070xt", TargetDetails{rdna4Wgp, &rx9070xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-9070.c4250
      .Case("rx9070", TargetDetails{rdna4Wgp, &rx9070Chip})
      // https://www.techpowerup.com/gpu-specs/radeon-ai-pro-r9700.c4290
      .Case("r9700", TargetDetails{rdna4Wgp, &r9700Chip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-9060-xt-16-gb.c4293
      .Case("rx9060xt", TargetDetails{rdna4Wgp, &rx9060xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941
      .Case("rx7900xtx", TargetDetails{rdna3Wgp, &rx7900xtxChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xt.c3912
      .Case("rx7900xt", TargetDetails{rdna3Wgp, &rx7900xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7800-xt.c3839
      .Case("rx7800xt", TargetDetails{rdna3Wgp, &rx7800xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7700-xt.c3911
      .Case("rx7700xt", TargetDetails{rdna3Wgp, &rx7700xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-pro-v710.c4234
      .Case("v710", TargetDetails{rdna3Wgp, &v710Chip})
      // https://www.techpowerup.com/gpu-specs/radeon-pro-w7900.c4147
      .Case("w7900", TargetDetails{rdna3Wgp, &w7900Chip})
      // https://www.techpowerup.com/gpu-specs/radeon-pro-w7800.c4148
      .Case("w7800", TargetDetails{rdna3Wgp, &w7800Chip})
      // https://www.techpowerup.com/gpu-specs/radeon-pro-w7700.c4184
      .Case("w7700", TargetDetails{rdna3Wgp, &w7700Chip})
      .Cases("rdna4", "gfx1200", "gfx1201", TargetDetails{rdna4Wgp, nullptr})
      .Cases("rdna3", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150",
             "gfx1151", TargetDetails{rdna3Wgp, nullptr})
      .Cases("rdna2", "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034",
             "gfx1035", "gfx1036", TargetDetails{rdna2Wgp, nullptr})
      .Cases("rdna1", "gfx1010", "gfx1011", "gfx1012", "gfx1013",
             TargetDetails{rdna1Wgp, nullptr})
      .Default(std::nullopt);
}

StringRef normalizeAMDGPUTarget(StringRef target) {
  if (target.starts_with("gfx"))
    return target;

  // We cannot accept rdnaN as a target for LLVM AMDGPU backend; so the
  // following is only meant for Vulkan but not HIP.
  if (target.starts_with("rdna"))
    return target;

  return llvm::StringSwitch<StringRef>(target.lower())
      .Cases("mi350x", "mi355x", "gfx950")
      .Cases("mi300a", "mi300x", "mi308x", "mi325x", "gfx942")
      .Cases("mi250x", "mi250", "mi210", "cdna2", "gfx90a")
      .Cases("mi100", "cdna1", "gfx908")
      .Cases("rx9070xt", "rx9070", "r9700", "gfx1201")
      .Case("rx9060xt", "gfx1200")
      .Cases("rx7900xtx", "rx7900xt", "w7900", "w7800", "gfx1100")
      .Cases("rx7800xt", "rx7700xt", "v710", "w7700", "gfx1101")
      .Default("");
}

//===----------------------------------------------------------------------===//
// Known Apple target details
//===----------------------------------------------------------------------===//

std::optional<TargetDetails> getAppleTargetDetails() {
  ComputeBitwidths computeBitwdiths =
      allIntComputeBits | ComputeBitwidths::FP32 | ComputeBitwidths::FP16;
  // clang-format off
  static const WgpDetails wgp = {
      computeBitwdiths,   allStorageBits,     allSubgroupOps,  allDotProductOps,
      /*mmaCount=*/0,     /*mmaOps=*/nullptr, /*scaledMmaCount=*/0,
      /*scaledMmaOps=*/nullptr,               {32, 32},
      {1024, 1024, 1024}, 1024,               32 * 1024,
      // Note: These values have not been checked and may be higher
      {0xffff, 0xffff, 0xffff}};
  // clang-format on

  return TargetDetails{&wgp, nullptr};
}

//===----------------------------------------------------------------------===//
// Known ARM target details
//===----------------------------------------------------------------------===//

const WgpDetails *getValhallWgpDetails() {
  // Recent drivers report support for shaderInt64. Aside from not widely
  // applicable, we don't know whether that's emulated and how performant it is.
  // So exclude that for now.
  ComputeBitwidths computeBitwdiths =
      ComputeBitwidths::Int32 | ComputeBitwidths::Int16 |
      ComputeBitwidths::Int8 | ComputeBitwidths::FP32 | ComputeBitwidths::FP16;
  // clang-format off
  static const WgpDetails valhallWgp = {
      computeBitwdiths,   allStorageBits,     allSubgroupOps,  allDotProductOps,
      /*mmaCount=*/0,     /*mmaOps=*/nullptr, /*scaledMmaCount=*/0,
      /*scaledMmaOps=*/nullptr,               {16, 16},        {512, 512, 512},
      512,                32 * 1024,
      // Note: These values have not been checked and may be higher
      {0xffff, 0xffff, 0xffff}};
  // clang-format on
  return &valhallWgp;
}

std::optional<TargetDetails> getARMGPUTargetDetails(StringRef target) {
  const WgpDetails *valhallWgp = getValhallWgpDetails();

  // Note that the underlying GPU may have certain capabilities but the Android
  // version and driver stack may not expose them. So the following is just and
  // will always be approximate.

  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      // Mali-G715: https://vulkan.gpuinfo.org/displayreport.php?id=29754
      .Cases("mali-g715", "mali-g615", "valhall4",
             TargetDetails{valhallWgp, nullptr})
      // Mali-G710: https://vulkan.gpuinfo.org/displayreport.php?id=30471
      .Cases("mali-g710", "mali-g510", "mali-g310", "valhall3",
             TargetDetails{valhallWgp, nullptr})
      // Mali-G78: https://vulkan.gpuinfo.org/displayreport.php?id=29994
      .Cases("mali-g78", "valhall2", TargetDetails{valhallWgp, nullptr})
      // Mali-G57: https://vulkan.gpuinfo.org/displayreport.php?id=24636
      .Cases("mali-g77", "mali-g57", "valhall1", "valhall",
             TargetDetails{valhallWgp, nullptr})
      .Default(std::nullopt);
}

StringRef normalizeARMGPUTarget(StringRef target) {
  if (target == "valhall")
    return "valhall1";
  if (target.starts_with("valhall"))
    return target;

  return llvm::StringSwitch<StringRef>(target.lower())
      .Cases("mali-g715", "mali-g615", "valhall4")
      .Cases("mali-g710", "mali-g510", "mali-g310", "valhall3")
      .Case("mali-78", "valhall2")
      .Cases("mali-g77", "mali-g57", "valhall1")
      .Default("");
}

//===----------------------------------------------------------------------===//
// Known NVIDIA target details
//===----------------------------------------------------------------------===//

// FIXME: In the following query functions, we are using AMD WMMA intrinsics
// that have different layout from NVIDIA WMMA intrinsics. This is fine given
// right now we only use this to indicate target features for Vulkan, where all
// cooperative matrix layouts are opaque. We need to create NVIDIA specific WMMA
// intrinsics if we need to have explicit layout analysis and register mapping.

const WgpDetails *getAmpereWgpDetails() {
  static const MMAIntrinsic mmaOps[] = {
      MMAIntrinsic::NV_WMMA_F32_16x16x16_F16,
      MMAIntrinsic::NV_WMMA_F16_16x16x16_F16,
  };
  static const WgpDetails ampereWgp = {allComputeBits,
                                       allStorageBits,
                                       allSubgroupOps,
                                       allDotProductOps,
                                       ARRAY_SIZE(mmaOps),
                                       mmaOps,
                                       0,
                                       nullptr,
                                       {32, 32},
                                       {1024, 1024, 1024},
                                       1024,
                                       163 * 1024,
                                       {0x7fffffff, 0xffff, 0xffff}};
  return &ampereWgp;
}

const WgpDetails *getTuringWgpDetails() {
  static const MMAIntrinsic mmaOps[] = {
      MMAIntrinsic::NV_WMMA_F32_16x16x16_F16,
      MMAIntrinsic::NV_WMMA_F16_16x16x16_F16,
  };
  static const WgpDetails turingWgp = {allComputeBits,
                                       allStorageBits,
                                       allSubgroupOps,
                                       allDotProductOps,
                                       ARRAY_SIZE(mmaOps),
                                       mmaOps,
                                       0,
                                       nullptr,
                                       {32, 32},
                                       {1024, 1024, 1024},
                                       1024,
                                       64 * 1024,
                                       {0x7fffffff, 0xffff, 0xffff}};
  return &turingWgp;
}

const WgpDetails *getVoltaWgpDetails() {
  static const MMAIntrinsic mmaOps[] = {
      MMAIntrinsic::NV_WMMA_F32_16x16x16_F16,
      MMAIntrinsic::NV_WMMA_F16_16x16x16_F16,
  };
  // clang-format off
  static const WgpDetails voltaWgp = {
      allComputeBits,       allStorageBits,     allSubgroupOps,
      DotProductOps::None,  ARRAY_SIZE(mmaOps), mmaOps,
      0,                    nullptr,            {32, 32},
      {1024, 1024, 1024},   1024,               96 * 1024,
      {0x7fffffff, 0xffff, 0xffff}};
  // clang-format on
  return &voltaWgp;
}

const WgpDetails *getPascalWgpDetails() {
  // clang-format off
  static const WgpDetails pascalWgp = {
      allComputeBits, allStorageBits, allSubgroupOps, DotProductOps::None,
      0, nullptr, // Pascal does not have tensor core support.
      0, nullptr,
      {32, 32}, {1024, 1024, 1024}, 1024, 48 * 1024,
      {0x7fffffff, 0xffff, 0xffff}};
  // clang-format on
  return &pascalWgp;
}

std::optional<TargetDetails> getNVIDIAGPUTargetDetails(StringRef target) {
  const WgpDetails *ampereWgp = getAmpereWgpDetails();
  const WgpDetails *turingWgp = getTuringWgpDetails();
  const WgpDetails *voltaWgp = getVoltaWgpDetails();
  const WgpDetails *pascalWgp = getPascalWgpDetails();

  static const ChipDetails a100Chip = {108};
  static const ChipDetails rtx3090tiChip = {84};
  static const ChipDetails rtx3090Chip = {82};
  static const ChipDetails rtx3080tiChip = {80};
  static const ChipDetails rtx3080Chip = {68};
  static const ChipDetails rtx3070tiChip = {48};
  static const ChipDetails rtx3070Chip = {46};

  // https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  // lists mappings from microarchitectures to compute capabilities.

  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      // https://www.techpowerup.com/gpu-specs/a100-sxm4-80-gb.c3746
      .Case("a100", TargetDetails{ampereWgp, &a100Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3090-ti.c3829
      .Case("rtx3090ti", TargetDetails{ampereWgp, &rtx3090tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
      .Case("rtx3090", TargetDetails{ampereWgp, &rtx3090Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3080-ti.c3735
      .Case("rtx3080ti", TargetDetails{ampereWgp, &rtx3080tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621
      .Case("rtx3080", TargetDetails{ampereWgp, &rtx3080Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3070-ti.c3675
      .Case("rtx3070ti", TargetDetails{ampereWgp, &rtx3070tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3070.c3674
      .Case("rtx3070", TargetDetails{ampereWgp, &rtx3070Chip})
      .Cases("ampere", "sm_80", "sm_86", "sm_87",
             TargetDetails{ampereWgp, nullptr})
      .Cases("turing", "sm_75", TargetDetails{turingWgp, nullptr})
      .Cases("volta", "sm_70", "sm_72", TargetDetails{voltaWgp, nullptr})
      .Cases("pascal", "sm_60", "sm_61", "sm_62",
             TargetDetails{pascalWgp, nullptr})
      .Default(std::nullopt);
}

StringRef normalizeNVIDIAGPUTarget(StringRef target) {
  if (target.starts_with("sm_"))
    return target;

  if (target.starts_with("rtx40"))
    return "sm_89";
  if (target.starts_with("rtx30"))
    return "sm_86";
  if (target.starts_with("rtx20"))
    return "sm_75";

  return llvm::StringSwitch<StringRef>(target.lower())
      .Case("a100", "sm_80")
      .Case("ampere", "sm_80") // Or sm_86/87; use smaller version
      .Case("turing", "sm_75")
      .Case("volta", "sm_70")  // Or sm_72; use smaller version
      .Case("pascal", "sm_60") // Or sm_61/62; use smaller version
      .Default("");
}

//===----------------------------------------------------------------------===//
// Known Qualcomm target details
//===----------------------------------------------------------------------===//

const WgpDetails *getAdrenoWgpDetails() {
  auto computeBitwdiths = ComputeBitwidths::Int32 | ComputeBitwidths::Int16 |
                          ComputeBitwidths::Int8 | ComputeBitwidths::FP32 |
                          ComputeBitwidths::FP16;
  auto storageBitwidths =
      StorageBitwidths::B64 | StorageBitwidths::B32 | StorageBitwidths::B16;
  // clang-format off
  static const WgpDetails adrenoWgp = {
      computeBitwdiths,   storageBitwidths,   allSubgroupOps,
      allDotProductOps,   /*mmaCount=*/0,     /*mmaOps=*/nullptr,
      /*scaledMmaCount=*/0,                   /*scaledMmaOps=*/nullptr,
      {64, 64},           {1024, 1024, 1024}, 1024,
      32 * 1024,
      // Note: These values have not been checked and may be higher
      {0xffff, 0xffff, 0xffff}};
  // clang-format on
  return &adrenoWgp;
}

bool verifyQualcommGPUTarget(StringRef target) {
  if (target == "adreno")
    return true;

  StringRef t = target;
  if (!t.consume_front("adreno-"))
    return false;

  // The can exist an optional L at the end.
  if (t.ends_with("l"))
    t = t.drop_back();

  // Check whether we have a product number
  unsigned number = 0;
  // StringRef::consumeInteger() returns true to signify errors.
  if (t.size() != 3 || t.consumeInteger(10, number))
    return false;

  return true;
}

std::optional<TargetDetails> getQualcommGPUTargetDetails(StringRef target) {
  const WgpDetails *adrenoWgp = getAdrenoWgpDetails();

  // Note that the underlying GPU may have certain capabilities but the Android
  // version and driver stack may not expose them. So the following is just and
  // will always be approximate.

  // Adreno GPUs are quite opaque regarding their generational information.
  // So right now we only have one target description for all cases.
  //
  // Though some example Adreno GPUs:
  // Adreno-750: https://vulkan.gpuinfo.org/displayreport.php?id=27414
  // Adreno-740: https://vulkan.gpuinfo.org/displayreport.php?id=19218
  // Adreno-730: https://vulkan.gpuinfo.org/displayreport.php?id=19382
  if (verifyQualcommGPUTarget(target))
    return TargetDetails{adrenoWgp, nullptr};

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Vulkan profile details
//===----------------------------------------------------------------------===//

const WgpDetails *getAndroidBaseline2022WgpDetails() {
  // The following details are from
  // https://github.com/KhronosGroup/Vulkan-Profiles/blob/main/profiles/VP_ANDROID_baseline_2022.json

  auto computeBitwdiths = ComputeBitwidths::Int32 | ComputeBitwidths::FP32;
  auto storageBitwidths = StorageBitwidths::B32;
  // FIXME: We cannot have a fixed subgroup size to target a profile; need to
  // have different targets for different subgroup sizes, or change CodeGen to
  // use symbolic subgroup size values, which can be hard for reduction.
  // It's kinda fine now given we don't allow any subgroup ops anyway here..

  // clang-format off
  static const WgpDetails androidWgp = {
      computeBitwdiths,    storageBitwidths,   SubgroupOps::None,
      DotProductOps::None, /*mmaCount=*/0,     /*mmaOps=*/nullptr,
      /*scaledMmaCount=*/0,                    /*scaledMmaOps=*/nullptr,
      {64, 64},            {128, 128, 64},     128,
      16 * 1024,
      {0xffff, 0xffff, 0xffff}};
  // clang-format on
  return &androidWgp;
}

std::optional<TargetDetails> getAndroidProfileDetails(StringRef target) {
  const WgpDetails *baseline2022Wgp = getAndroidBaseline2022WgpDetails();

  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      .Case("vp_android_baseline_2022", TargetDetails{baseline2022Wgp, nullptr})
      .Default(std::nullopt);
}

} // namespace

//===----------------------------------------------------------------------===//
// Query functions
//===----------------------------------------------------------------------===//

std::optional<L1CacheInfo> getL1CacheInfo(TargetAttr target) {
  // TODO(kuhar): Add L1 cache query for other HIP targets.
  if (!target || !llvm::is_contained({"gfx90a", "gfx942"}, target.getArch())) {
    return std::nullopt;
  }
  return L1CacheInfo{/*cacheLineBytes=*/128, /*cacheSets=*/4};
}

TargetAttr getMetalTargetDetails(MLIRContext *context) {
  return createTargetAttr(*getAppleTargetDetails(), /*arch=*/"apple",
                          /*features=*/"spirv:v1.3,cap:Shader", context);
}

TargetAttr getCUDATargetDetails(StringRef target, StringRef features,
                                MLIRContext *context) {
  if (std::optional<TargetDetails> details = getNVIDIAGPUTargetDetails(target))
    return createTargetAttr(*details, normalizeNVIDIAGPUTarget(target),
                            features, context);
  return nullptr;
}

StringRef normalizeCUDATarget(StringRef target) {
  return normalizeNVIDIAGPUTarget(target);
}

TargetAttr getHIPTargetDetails(StringRef target, StringRef features,
                               MLIRContext *context) {
  if (std::optional<TargetDetails> details = getAMDGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeAMDGPUTarget(target), features,
                            context);
  }
  return nullptr;
}

Attribute getHIPTargetEncodingLayoutAttr(TargetAttr target,
                                         StringRef resolver) {
  if (resolver == kDataTilingEncodingLayoutResolverName) {
    // Return a GPUEncodingResolverAttr with an empty configuration. The
    // addtional attributes will be attached by the `cloneWithSimplifiedConfig`
    // interface method when the resolver needs to be configured.
    return IREE::GPU::GPUEncodingResolverAttr::get(target.getContext(), {});
  }

  if (resolver == kPadEncodingLayoutResolverName) {
    return IREE::GPU::GPUPaddingResolverAttr::get(
        target.getContext(),
        /*cache_line_bytes=*/std::nullopt,
        /*cache_sets=*/std::nullopt);
  }
  return nullptr;
}

StringRef normalizeHIPTarget(StringRef target) {
  return normalizeAMDGPUTarget(target);
}

TargetAttr getVulkanTargetDetails(llvm::StringRef target,
                                  MLIRContext *context) {
  // Go through each vendor's target details. This assumes we won't have
  // duplicated product or microarchitecture names among vendors, which should
  // be the case.

  // TODO: Add feature bits for physical storage buffer.

  if (std::optional<TargetDetails> details = getAMDGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeAMDGPUTarget(target),
                            /*features=*/"spirv:v1.6,cap:Shader", context);
  }
  if (std::optional<TargetDetails> details = getARMGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeARMGPUTarget(target),
                            /*features=*/"spirv:v1.6,cap:Shader", context);
  }
  if (std::optional<TargetDetails> details =
          getNVIDIAGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeNVIDIAGPUTarget(target),
                            /*features=*/"spirv:v1.6,cap:Shader", context);
  }
  if (std::optional<TargetDetails> details =
          getQualcommGPUTargetDetails(target)) {
    return createTargetAttr(*details, target,
                            /*features=*/"spirv:v1.6,cap:Shader", context);
  }

  // Go through common profiles if not hit in the above.

  if (std::optional<TargetDetails> details = getAndroidProfileDetails(target)) {
    return createTargetAttr(*details, target,
                            /*features=*/"spirv:v1.3,cap:Shader", context);
  }
  return nullptr;
}

TargetAttr getWebGPUTargetDetails(MLIRContext *context) {
  // TODO(scotttodd): find list of SPIR-V capabilities and extensions supported
  // by WebGPU/WGSL.
  auto computeBitwdiths = ComputeBitwidths::Int32 | ComputeBitwidths::FP32;
  auto storageBitwidths = StorageBitwidths::B32;
  // clang-format off
  static const WgpDetails wgp = {
      computeBitwdiths,    storageBitwidths,   SubgroupOps::None,
      DotProductOps::None, /*mmaCount=*/0,     /*mmaOps=*/nullptr,
      /*scaledMmaCount=*/0,                    /*scaledMmaOps=*/nullptr,
      {32, 32},            {128, 128, 64},     128,
      16 * 1024,
      {0xffff, 0xffff, 0xffff}};
  // clang-format on

  return createTargetAttr({&wgp, nullptr}, /*arch=*/"",
                          "spirv:v1.0,cap:Shader,ext:SPV_KHR_storage_buffer_"
                          "storage_class,ext:SPV_KHR_non_semantic_info",
                          context);
}

TargetAttr getFullTarget(StringRef targetAPI, StringRef aliasTarget,
                         StringRef features, MLIRContext *context) {
  return llvm::StringSwitch<TargetAttr>(targetAPI)
      .Case("cuda", getCUDATargetDetails(aliasTarget, features, context))
      .Case("hip", getHIPTargetDetails(aliasTarget, features, context))
      .Case("vulkan", getVulkanTargetDetails(aliasTarget, context))
      .Default(nullptr);
}

} // namespace mlir::iree_compiler::IREE::GPU
