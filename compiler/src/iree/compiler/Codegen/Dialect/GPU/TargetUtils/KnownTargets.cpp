// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"

#include <optional>
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/StringSwitch.h"

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

// Core level feature/limit details
struct CoreDetails {
  ComputeBitwidths compute;
  StorageBitwidths storage;
  SubgroupOps subgroupOps;
  DotProductOps dotproductOps;
  uint32_t mmaCount;
  const MMAIntrinsic *mmaOps;

  std::array<int32_t, 2> subgroupSizeChoices;
  std::array<int32_t, 3> maxWorkgroupSizes;
  uint32_t maxThreadSize;
  uint32_t maxWorkgroupMemoryBytes;
};

// Chip level feature/limit details
struct ChipDetails {
  uint32_t coreCount;
};

// Full target details
struct TargetDetails {
  const CoreDetails *core;
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
TargetAttr createTargetAttr(const TargetDetails &details, TargetAPI api,
                            StringRef arch, MLIRContext *context) {
  const CoreDetails *core = details.core;

  SmallVector<MMAAttr, 8> mmaAttrs;
  mmaAttrs.reserve(core->mmaCount);
  for (int i = 0; i < core->mmaCount; ++i)
    mmaAttrs.push_back(MMAAttr::get(context, core->mmaOps[i]));

  SmallVector<int32_t, 2> subgroupSizes;
  subgroupSizes.push_back(core->subgroupSizeChoices.front());
  if (core->subgroupSizeChoices.back() != core->subgroupSizeChoices.front())
    subgroupSizes.push_back(core->subgroupSizeChoices.back());

  auto targetCore = TargetCoreAttr::get(
      context, ComputeBitwidthsAttr::get(context, details.core->compute),
      StorageBitwidthsAttr::get(context, core->storage),
      SubgroupOpsAttr::get(context, core->subgroupOps),
      DotProductOpsAttr::get(context, core->dotproductOps),
      MMAOpsArrayAttr::get(context, mmaAttrs),
      DenseI32ArrayAttr::get(context, subgroupSizes),
      DenseI32ArrayAttr::get(context, core->maxWorkgroupSizes),
      core->maxThreadSize, core->maxWorkgroupMemoryBytes);

  TargetChipAttr targetChip;
  if (details.chip)
    targetChip = TargetChipAttr::get(context, details.chip->coreCount);

  return TargetAttr::get(context, api, arch, targetCore, targetChip);
}

//===----------------------------------------------------------------------===//
// Known AMD target details
//===----------------------------------------------------------------------===//

const CoreDetails *getCDNA3CoreDetails() {
  static const MMAIntrinsic cdna3MMAOps[] = {
      MMAIntrinsic::MFMA_F16_16x16x16_F32,
      MMAIntrinsic::MFMA_F16_32x32x8_F32,
  };
  static const CoreDetails cdna3Core = {
      allComputeBits,   allStorageBits,          allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(cdna3MMAOps), cdna3MMAOps,
      {64, 64},         {1024, 1024, 1024},      1024,
      64 * 1024};
  return &cdna3Core;
}

const CoreDetails *getCDNA2CoreDetails() {
  static const MMAIntrinsic cdna2MMAOps[] = {
      MMAIntrinsic::MFMA_F16_16x16x16_F32,
      MMAIntrinsic::MFMA_F16_32x32x8_F32,
  };
  static const CoreDetails cdna2Core = {
      allComputeBits,   allStorageBits,          allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(cdna2MMAOps), cdna2MMAOps,
      {64, 64},         {1024, 1024, 1024},      1024,
      64 * 1024};
  return &cdna2Core;
}

const CoreDetails *getCDNA1CoreDetails() {
  static const MMAIntrinsic cdna1MMAOps[] = {
      MMAIntrinsic::MFMA_F16_16x16x16_F32,
      MMAIntrinsic::MFMA_F16_32x32x8_F32,
  };
  static const CoreDetails cdna1Core = {
      allComputeBits,   allStorageBits,          allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(cdna1MMAOps), cdna1MMAOps,
      {64, 64},         {1024, 1024, 1024},      1024,
      64 * 1024};
  return &cdna1Core;
}

const CoreDetails *getRDNA3CoreDetails() {
  static const MMAIntrinsic rdna3MMAOps[] = {
      MMAIntrinsic::WMMA_F16_16x16x16_F32,
  };
  static const CoreDetails rdna3Core = {
      allComputeBits,   allStorageBits,          allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(rdna3MMAOps), rdna3MMAOps,
      {32, 64},         {1024, 1024, 1024},      1024,
      64 * 1024};
  return &rdna3Core;
}

std::optional<TargetDetails> getAMDGPUTargetDetails(StringRef target) {
  const CoreDetails *cdna3Core = getCDNA3CoreDetails();
  const CoreDetails *cdna2Core = getCDNA2CoreDetails();
  const CoreDetails *cdna1Core = getCDNA1CoreDetails();
  const CoreDetails *rdna3Core = getRDNA3CoreDetails();

  // "AMD Instinct MI300 Series Product Offerings" in Page 23 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
  static const ChipDetails mi300xChip = {304};
  static const ChipDetails mi300aChip = {228};

  // "AMD Instinct MI200 Series Accelerator Product Offerings" in Page 14 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
  static const ChipDetails mi250xChip = {220};
  static const ChipDetails mi250Chip = {208};
  static const ChipDetails mi210Chip = {104};

  // "AMD CDNA Architecture Compute Units" in Page 5 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
  static const ChipDetails mi100Chip = {120};

  static const ChipDetails rx7900xtxChip = {96};
  static const ChipDetails rx7900xtChip = {84};
  static const ChipDetails rx7800xtChip = {60};
  static const ChipDetails rx7700xtChip = {54};

  // See https://llvm.org/docs/AMDGPUUsage.html#processors for gfxN to
  // cdnaN/rdnaN mapping.

  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      .Case("mi300x", TargetDetails{cdna3Core, &mi300xChip})
      .Case("mi300a", TargetDetails{cdna3Core, &mi300aChip})
      .Cases("cdna3", "gfx940", "gfx941", "gfx942",
             TargetDetails{cdna3Core, nullptr})
      .Case("mi250x", TargetDetails{cdna2Core, &mi250xChip})
      .Case("mi250", TargetDetails{cdna2Core, &mi250Chip})
      .Case("mi210", TargetDetails{cdna2Core, &mi210Chip})
      .Cases("cdna2", "gfx90a", TargetDetails{cdna2Core, nullptr})
      .Case("mi100", TargetDetails{cdna1Core, &mi100Chip})
      .Cases("cdna1", "gfx908", TargetDetails{cdna1Core, nullptr})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941
      .Case("rx7900xtx", TargetDetails{rdna3Core, &rx7900xtxChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xt.c3912
      .Case("rx7900xt", TargetDetails{rdna3Core, &rx7900xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7800-xt.c3839
      .Case("rx7800xt", TargetDetails{rdna3Core, &rx7800xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7700-xt.c3911
      .Case("rx7700xt", TargetDetails{rdna3Core, &rx7700xtChip})
      .Cases("rdna3", "gfx1100", "gfx1101", "gfx1102", "gfx1103",
             TargetDetails{rdna3Core, nullptr})
      .Default(std::nullopt);
}

StringRef normalizeAMDGPUTarget(StringRef target) {
  if (target.starts_with("gfx"))
    return target;

  return llvm::StringSwitch<StringRef>(target.lower())
      .Case("mi300x", "gfx942")
      .Case("mi300a", "gfx940")
      .Cases("mi250x", "mi250", "mi210", "cdna2", "gfx90a")
      .Cases("rx7900xtx", "rx7900xt", "gfx1100")
      .Cases("rx7800xt", "rx7700xt", "gfx1101")
      .Default(StringRef());
}

//===----------------------------------------------------------------------===//
// Known NVIDIA target details
//===----------------------------------------------------------------------===//

const CoreDetails *getAmpereCoreDetails() {
  static const CoreDetails ampereCore = {
      allComputeBits, allStorageBits,     allSubgroupOps, allDotProductOps, 0,
      nullptr, // TODO: Add tensor core operations
      {32, 32},       {1024, 1024, 1024}, 1024,           163 * 1024};
  return &ampereCore;
}

const CoreDetails *getTuringCoreDetails() {
  static const CoreDetails turingCore = {
      allComputeBits, allStorageBits,     allSubgroupOps, allDotProductOps, 0,
      nullptr, // TODO: Add tensor core operations
      {32, 32},       {1024, 1024, 1024}, 1024,           64 * 1024};
  return &turingCore;
}

const CoreDetails *getVoltaCoreDetails() {
  // clang-format off
  static const CoreDetails voltaCore = {
      allComputeBits, allStorageBits, allSubgroupOps, DotProductOps::None,
      0, nullptr, // TODO: Add tensor core operations
      {32, 32}, {1024, 1024, 1024}, 1024, 96 * 1024};
  // clang-format on
  return &voltaCore;
}

const CoreDetails *getPascalCoreDetails() {
  // clang-format off
  static const CoreDetails pascalCore = {
      allComputeBits, allStorageBits, allSubgroupOps, DotProductOps::None,
      0, nullptr, // Pascal does not have tensor core support.
      {32, 32}, {1024, 1024, 1024}, 1024, 48 * 1024};
  // clang-format on
  return &pascalCore;
}

std::optional<TargetDetails> getNVIDIAGPUTargetDetails(StringRef target) {
  const CoreDetails *ampereCore = getAmpereCoreDetails();
  const CoreDetails *turingCore = getTuringCoreDetails();
  const CoreDetails *voltaCore = getVoltaCoreDetails();
  const CoreDetails *pascalCore = getPascalCoreDetails();

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
      .Case("a100", TargetDetails{ampereCore, &a100Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3090-ti.c3829
      .Case("rtx3090ti", TargetDetails{ampereCore, &rtx3090tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
      .Case("rtx3090", TargetDetails{ampereCore, &rtx3090Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3080-ti.c3735
      .Case("rtx3080ti", TargetDetails{ampereCore, &rtx3080tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621
      .Case("rtx3080", TargetDetails{ampereCore, &rtx3080Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3070-ti.c3675
      .Case("rtx3070ti", TargetDetails{ampereCore, &rtx3070tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3070.c3674
      .Case("rtx3070", TargetDetails{ampereCore, &rtx3070Chip})
      .Cases("ampere", "sm_80", "sm_86", "sm_87",
             TargetDetails{ampereCore, nullptr})
      .Cases("turing", "sm_75", TargetDetails{turingCore, nullptr})
      .Cases("volta", "sm_70", "sm_72", TargetDetails{voltaCore, nullptr})
      .Cases("pascal", "sm_60", "sm_61", "sm_62",
             TargetDetails{pascalCore, nullptr})
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
      .Default(StringRef());
}

} // namespace

//===----------------------------------------------------------------------===//
// Query functions
//===----------------------------------------------------------------------===//

TargetAttr getHIPTargetDetails(StringRef target, MLIRContext *context) {
  if (auto details = getAMDGPUTargetDetails(target)) {
    return createTargetAttr(*details, TargetAPI::HIP,
                            normalizeAMDGPUTarget(target), context);
  }
  return nullptr;
}

StringRef normalizeHIPTarget(StringRef target) {
  return normalizeAMDGPUTarget(target);
}

TargetAttr getCUDATargetDetails(StringRef target, MLIRContext *context) {
  if (auto details = getNVIDIAGPUTargetDetails(target))
    return createTargetAttr(*details, TargetAPI::CUDA,
                            normalizeNVIDIAGPUTarget(target), context);
  return nullptr;
}

StringRef normalizeCUDATarget(StringRef target) {
  return normalizeNVIDIAGPUTarget(target);
}

bool isKnownAbbrTarget(AbbrTargetAttr abbrTarget) {
  switch (abbrTarget.getApi()) {
  case TargetAPI::CUDA:
    return getNVIDIAGPUTargetDetails(abbrTarget.getChip()).has_value();
  case TargetAPI::HIP:
    return getAMDGPUTargetDetails(abbrTarget.getChip()).has_value();
  default:
    break;
  }
  return false;
}

TargetAttr getFullTarget(AbbrTargetAttr abbrTarget) {
  MLIRContext *context = abbrTarget.getContext();
  switch (abbrTarget.getApi()) {
  case TargetAPI::CUDA:
    return getCUDATargetDetails(abbrTarget.getChip(), context);
  case TargetAPI::HIP:
    return getHIPTargetDetails(abbrTarget.getChip(), context);
  default:
    break;
  }
  return nullptr;
}

} // namespace mlir::iree_compiler::IREE::GPU
