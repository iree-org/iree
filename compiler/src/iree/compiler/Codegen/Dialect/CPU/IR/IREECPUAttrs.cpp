// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/MMAUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::CPU {

//===----------------------------------------------------------------------===//
// CPU Pipeline Attribute
//===----------------------------------------------------------------------===//

static CPUPipelineBuilder &getCPUPipelineBuilderStorage() {
  static CPUPipelineBuilder builder = nullptr;
  return builder;
}

void registerCPUPipelineBuilder(CPUPipelineBuilder builder) {
  // Expected to be called exactly once during global init, so thread
  // safety is not a concern.
  [[maybe_unused]] static bool registered = false;
  assert(!registered && "CPU pipeline builder registered more than once");
  registered = true;
  getCPUPipelineBuilderStorage() = builder;
}

LogicalResult
PipelineAttr::buildPipeline(OpPassManager &pm,
                            const CodegenPipelineOptions *options) const {
  CPUPipelineBuilder builder = getCPUPipelineBuilderStorage();
  assert(builder && "no CPU pipeline builder registered; ensure "
                    "registerCodegenLLVMCPUPasses() was called");
  return builder(*this, pm, options);
}

//===----------------------------------------------------------------------===//
// CPU Specific Lowering Config Attributes
//===----------------------------------------------------------------------===//

constexpr StringLiteral kDistributionConfigKey = "distribution";
constexpr StringLiteral kCacheParallelConfigKey = "cache_parallel";
constexpr StringLiteral kCacheReductionConfigKey = "cache_reduction";
constexpr StringLiteral kVectorCommonParallelConfigKey =
    "vector_common_parallel";
constexpr StringLiteral kVectorReductionConfigKey = "vector_reduction";
constexpr StringLiteral kVectorInnerParallelConfigKey = "vector_inner_parallel";

SmallVector<int> getTilingLevelsAsInts() {
  return llvm::to_vector(
      llvm::seq<int>(0, llvm::to_underlying(TilingLevel::MaxNumTileLevels)));
}

/// Returns the entry key for the config in IREE::CPU::LoweringConfigAttr.
/// Returns null if `level` is invalid.
StringRef getTilingLevelName(TilingLevel level) {
  switch (level) {
  case TilingLevel::DistributionTiles:
    return kDistributionConfigKey;
  case TilingLevel::CacheParallelTiles:
    return kCacheParallelConfigKey;
  case TilingLevel::CacheReductionTiles:
    return kCacheReductionConfigKey;
  case TilingLevel::VectorCommonParallelTiles:
    return kVectorCommonParallelConfigKey;
  case TilingLevel::VectorReductionTiles:
    return kVectorReductionConfigKey;
  case TilingLevel::VectorInnerParallelTiles:
    return kVectorInnerParallelConfigKey;
  case TilingLevel::MaxNumTileLevels:
  case TilingLevel::InvalidLevel:
  default:
    return StringRef();
  }
}

static SmallVector<int64_t> getTileSizes(DictionaryAttr config,
                                         TilingLevel level) {
  auto attr = config.getAs<IREE::Codegen::LoweringConfigTilingLevelAttr>(
      getTilingLevelName(level));
  if (!attr) {
    return {};
  }
  return SmallVector<int64_t>(attr.getSizes());
}

/// Returns true if the given entry `key` should use
/// IREE::Codegen::LoweringConfigTilingLevelAttr as value type.
static bool isLoweringConfigTilingLevelKey(StringRef key) {
  SmallVector<StringLiteral> allowList = {
      kDistributionConfigKey,         kCacheParallelConfigKey,
      kCacheReductionConfigKey,       kVectorReductionConfigKey,
      kVectorCommonParallelConfigKey, kVectorInnerParallelConfigKey};
  return llvm::is_contained(allowList, key);
}

LogicalResult
LoweringConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           DictionaryAttr config) {
  for (NamedAttribute attr : config) {
    if (isLoweringConfigTilingLevelKey(attr.getName()) &&
        !isa<IREE::Codegen::LoweringConfigTilingLevelAttr>(attr.getValue())) {
      return emitError() << attr.getName()
                         << " is not LoweringConfigTilingLevelAttr: "
                         << attr.getValue();
    }
  }
  return success();
}

Attribute LoweringConfigAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess()) {
    return {};
  }

  MLIRContext *ctx = parser.getContext();
  SmallVector<NamedAttribute> dictItems;
  bool first = true;
  while (parser.parseOptionalGreater()) {
    if (!first && parser.parseComma()) {
      return {};
    }
    first = false;

    std::string keyStr;
    if (parser.parseKeywordOrString(&keyStr) || parser.parseEqual()) {
      return {};
    }
    StringAttr key = StringAttr::get(ctx, keyStr);
    Attribute value;
    if (isLoweringConfigTilingLevelKey(keyStr)) {
      value = IREE::Codegen::LoweringConfigTilingLevelAttr::parse(parser, type);
      if (!value) {
        return {};
      }
    } else {
      if (parser.parseAttribute(value)) {
        return {};
      }
    }
    dictItems.emplace_back(key, value);
  }
  auto dictAttr = DictionaryAttr::get(ctx, dictItems);
  return parser.getChecked<LoweringConfigAttr>(ctx, dictAttr);
}

void LoweringConfigAttr::print(AsmPrinter &printer) const {
  printer << "<";
  auto dictAttr = getConfig();
  llvm::interleaveComma(dictAttr, printer, [&](NamedAttribute attr) {
    // Use `.str()` to avoid wrapping the key string with `"`.
    printer << attr.getName().str() << " = ";
    if (isLoweringConfigTilingLevelKey(attr.getName())) {
      // Do not use `printAttribute` that avoids printing dialect namespace as
      // prefix.
      cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(attr.getValue())
          .print(printer);
    } else {
      printer.printAttribute(attr.getValue());
    }
  });
  printer << ">";
}

LoweringConfigAttr LoweringConfigAttr::get(MLIRContext *ctx,
                                           SmallVector<NamedAttribute> items) {
  return Base::get(ctx, DictionaryAttr::get(ctx, items));
}

Attribute LoweringConfigAttr::getTilingLevelAttr(MLIRContext *ctx,
                                                 ArrayRef<int64_t> tileSizes) {
  return IREE::Codegen::LoweringConfigTilingLevelAttr::get(
      ctx, tileSizes, /*interchange=*/{}, /*scalableFlags=*/{});
}

Attribute LoweringConfigAttr::getTilingLevelAttr(MLIRContext *ctx,
                                                 ArrayRef<int64_t> tileSizes,
                                                 ArrayRef<bool> scalableFlags) {
  return IREE::Codegen::LoweringConfigTilingLevelAttr::get(
      ctx, tileSizes, /*interchange=*/{}, scalableFlags);
}

SmallVector<LoweringConfigLevelInfo>
LoweringConfigAttr::getAvailableTilingInfo() {
  SmallVector<LoweringConfigLevelInfo> result;
  for (int i : IREE::CPU::getTilingLevelsAsInts()) {
    if (!hasTilingLevel(i)) {
      continue;
    }
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        getTilingLevelAttr(i));
    LoweringConfigLevelInfo item;
    item.level = static_cast<TilingLevel>(i);
    llvm::append_range(item.sizes, attr.getSizes());
    llvm::append_range(item.scalableFlags, attr.getScalableFlags());
    result.push_back(item);
  }
  return result;
}

SmallVector<int64_t> LoweringConfigAttr::getWorkgroupTileSizes() const {
  return getTileSizes(getConfig(), TilingLevel::DistributionTiles);
}

SmallVector<OpFoldResult>
LoweringConfigAttr::getTilingLevelSizes(OpBuilder &builder, unsigned level,
                                        Operation *op) const {
  assert(level < llvm::to_underlying(TilingLevel::MaxNumTileLevels) &&
         "invalid level");
  return llvm::map_to_vector(
      getTileSizes(getConfig(), static_cast<TilingLevel>(level)),
      [&](int64_t t) -> OpFoldResult { return builder.getIndexAttr(t); });
}

bool LoweringConfigAttr::hasTilingLevel(unsigned level) const {
  return getConfig().contains(
      getTilingLevelName(static_cast<TilingLevel>(level)));
}

bool LoweringConfigAttr::hasWorkgroupTilingLevel() const {
  return getConfig().contains(
      getTilingLevelName(TilingLevel::DistributionTiles));
}

std::optional<unsigned> LoweringConfigAttr::getNumTilingLevels() const {
  return llvm::count_if(getConfig(), [](NamedAttribute attr) {
    return isLoweringConfigTilingLevelKey(attr.getName());
  });
}

SmallVector<int64_t>
LoweringConfigAttr::getStaticTilingLevelSizes(unsigned level,
                                              Operation *) const {
  assert(level < llvm::to_underlying(TilingLevel::MaxNumTileLevels) &&
         "invalid level");
  return getTileSizes(getConfig(), static_cast<TilingLevel>(level));
}

Attribute LoweringConfigAttr::getTilingLevelAttr(unsigned level) const {
  assert(level < llvm::to_underlying(TilingLevel::MaxNumTileLevels) &&
         "invalid level");
  StringRef key = getTilingLevelName(static_cast<TilingLevel>(level));
  DictionaryAttr config = getConfig();
  if (!config || !config.contains(key)) {
    return {};
  }
  return config.get(key);
}

constexpr std::array vectorTilingLevels{TilingLevel::VectorCommonParallelTiles,
                                        TilingLevel::VectorReductionTiles,
                                        TilingLevel::VectorInnerParallelTiles};

std::optional<SmallVector<int64_t>> LoweringConfigAttr::getVectorSizes() const {
  SmallVector<int64_t> result;
  for (TilingLevel level : vectorTilingLevels) {
    if (!hasTilingLevel(llvm::to_underlying(level))) {
      continue;
    }
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        getTilingLevelAttr(llvm::to_underlying(level)));
    if (result.empty()) {
      result.resize(attr.getSizes().size(), 0);
    }
    for (auto [idx, size] : llvm::enumerate(attr.getSizes())) {
      if (size == 0) {
        continue;
      }
      if (result[idx] != 0) {
        return std::nullopt;
      }
      result[idx] = size;
    }
  }
  return result;
}

SmallVector<bool> LoweringConfigAttr::getVectorScalableFlags() const {
  SmallVector<bool> result;
  for (TilingLevel level : vectorTilingLevels) {
    if (!hasTilingLevel(llvm::to_underlying(level))) {
      continue;
    }
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        getTilingLevelAttr(llvm::to_underlying(level)));
    ArrayRef<bool> scalableFlags = attr.getScalableFlags();
    if (result.empty() && !scalableFlags.empty()) {
      result.resize(attr.getSizes().size(), false);
    }
    for (auto [idx, flag] : llvm::enumerate(scalableFlags)) {
      result[idx] |= flag;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// CPU MMA intrinsic layout (MxNxK shape and element types)
//===----------------------------------------------------------------------===//

// Helper function for getIntrinsicSwizzle.
// Allows succinctly specifying the tiles for the common case of row-major tiles
// i.e. when the TileSwizzle merely encodes the 2D tile shape and no further
// expand/transpose.
// This case is very common for CPU MMA intrinsics due to:
// 1. CPU matmul instructions having simple tile layouts.
// 2. The inner_tiled op RHS being transposed, meaning that the row-major RHS
//    tile really is a column-major RHS matmul tile.
//
// If you're trying to add support for a new intrinsic that doesn't have a
// row-major tile layout, don't look at this function, go directly to
// getIntrinsicSwizzle.
std::optional<std::tuple<int64_t, int64_t, int64_t>>
getRowMajorTilesMNKShape(MMAIntrinsic intrinsic) {
  using Tuple = std::tuple<int64_t, int64_t, int64_t>;
  switch (intrinsic) {
  case MMAIntrinsic::None:
    return Tuple{0, 0, 0};
  case MMAIntrinsic::MMA_X86_AVX2_FMA_1x8x1_F32_F32:
    return Tuple{1, 8, 1};
  case MMAIntrinsic::MMA_X86_AVX2_FMA_8x1x1_F32_F32:
    return Tuple{8, 1, 1};
  case MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64:
    return Tuple{1, 8, 1};
  case MMAIntrinsic::MMA_X86_AVX512_8x1x1_F64_F64:
    return Tuple{8, 1, 1};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
    return Tuple{1, 16, 1};
  case MMAIntrinsic::MMA_X86_AVX512_16x1x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x1_F32_F16_CASTF32:
    return Tuple{16, 1, 1};
  case MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16:
    return Tuple{1, 32, 1};
  case MMAIntrinsic::MMA_X86_AVX512FP16_32x1x1_F16_F16:
    return Tuple{32, 1, 1};
  case MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
    return Tuple{1, 16, 2};
  case MMAIntrinsic::MMA_X86_AVX512BF16_16x1x2_F32_BF16:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x2_I32_I8_CASTI16:
    return Tuple{16, 1, 2};
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x4_I32_UI8_I8:
    return Tuple{1, 16, 4};
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x4_I32_I8_UI8:
    return Tuple{16, 1, 4};
  // 16x16x2: the LHS and RHS tiles *are* row-major (16x2 i8 panels); only the
  // ACC tile has a non-row-major layout, hand-rolled in `getIntrinsicSwizzle`.
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16:
    return Tuple{16, 16, 2};
  default:
    if (isGenericScalar(intrinsic)) {
      return Tuple{1, 1, 1};
    }
    return {};
  }
}

// Bit-layout constants for the `MMAIntrinsic` enum value (see IREECPUEnums.td
// for the 0xABCD scheme). The high byte (`kMMAIntrinsicISAMask`) encodes the
// architecture (nibble A) and the ISA-extension family (nibble B) together
// — the abbreviation is spelled `ISA` (uppercase) here to avoid reading as
// the English "is a". The generic-scalar family uses A=F, B=0; the low byte
// then holds the register-budget heuristic value.
constexpr uint32_t kMMAIntrinsicISAMask = 0xFF00;
constexpr uint32_t kMMAIntrinsicISAGeneric = 0xF000;
constexpr uint32_t kMMAIntrinsicGenericBudgetMask = 0x00FF;
constexpr uint32_t kMMAIntrinsicISAX86Avx2 = 0x1200;
constexpr uint32_t kMMAIntrinsicISAX86Avx512 = 0x1300;
constexpr uint32_t kMMAIntrinsicISAArmSve = 0x2200;

bool isGenericScalar(MMAIntrinsic intr) {
  return (static_cast<uint32_t>(intr) & kMMAIntrinsicISAMask) ==
         kMMAIntrinsicISAGeneric;
}

int64_t getGenericScalarRegisterBudget(MMAIntrinsic intr) {
  assert(isGenericScalar(intr));
  return static_cast<uint32_t>(intr) & kMMAIntrinsicGenericBudgetMask;
}

int64_t getRegisterSpaceBytes(MMAIntrinsic intrinsic) {
  // Total architectural vector register file size, in bytes. The inner-tiled
  // cost model uses this as the capacity for the union of the ACC, LHS and
  // RHS tiles. For scalable ISAs we treat the vector length as its minimum
  // (1 × 128 bits = 16 bytes per register); this is a deliberate
  // simplification — the resulting `intrinsics_m`/`intrinsics_n` choices are
  // good enough in practice and avoid propagating scalability into the cost
  // model.
  uint32_t isa = static_cast<uint32_t>(intrinsic) & kMMAIntrinsicISAMask;
  switch (isa) {
  case kMMAIntrinsicISAX86Avx2: // 16 YMM × 32 B.
    return 16 * 32;
  case kMMAIntrinsicISAX86Avx512: // 32 ZMM × 64 B.
    return 32 * 64;
  case kMMAIntrinsicISAArmSve: // 32 Z × (VL treated as 128 bits).
    return 32 * 16;
  default:
    // Plausible default, but override it on each arch you care for.
    return 16 * 32;
  }
}

/// Helper for `getIntrinsicSwizzle`:
/// Ensures every `expandShape` row has at least one piece (unit
/// dims that received no explicit `expand` get a non-scalable size-1 Internal
/// piece), and sets `permutation` to the identity over the total number of
/// expanded dims.
static Codegen::TileSwizzle fixupSwizzle(Codegen::TileSwizzle swizzle) {
  for (auto &group : swizzle.expandShape()) {
    if (group.empty()) {
      group.push_back(Codegen::TileSwizzle::Dim::internal(1));
    }
  }
  auto &permutation = swizzle.permutation();
  permutation.resize(swizzle.getExpandedSize());
  for (size_t i = 0; i < permutation.size(); ++i) {
    permutation[i] = i;
  }
  return swizzle;
}

Codegen::TileSwizzle getIntrinsicSwizzle(IREE::CPU::MMAIntrinsic mma,
                                         int operandIdx) {
  using TileSwizzle = Codegen::TileSwizzle;
  using Dim = TileSwizzle::Dim;

  // SVE has scalable dims that the row-major path below can't express, so we
  // hand-roll the swizzle. The natural orientation is 1×4VL×1: 4VL lives on
  // RHS dim 0 (N) and ACC dim 0 (the convention is `(N, M)` here). The
  // transposed orientation 4VL×1×1 mirrors that: 4VL on LHS dim 0 (M) and
  // ACC dim 0 (now `(M, N)`).
  if (mma == MMAIntrinsic::MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32 ||
      mma == MMAIntrinsic::MMA_ARM_SVE_FMLA_4VLx1x1_F32_F32) {
    bool transposed = (mma == MMAIntrinsic::MMA_ARM_SVE_FMLA_4VLx1x1_F32_F32);
    TileSwizzle swizzle;
    swizzle.expandShape().resize(2);
    int operandWith4VL = transposed ? 0 : 1;
    if (operandIdx == operandWith4VL || operandIdx == 2) {
      Codegen::expand(
          swizzle, /*srcDim=*/0,
          Dim::internal(4, Dim::SymbolicMultiplier::ArmSveVLIn128bitUnits));
    }
    return fixupSwizzle(std::move(swizzle));
  }

  // 16x16x2 i8: LHS/RHS are plain row-major 16x2 i8 panels (handled by the
  // row-major path below), but the 16x16 i32 ACC tile uses a block-interleaved
  // layout that mirrors the mmt4d ukernel's in-register `acc[4][4]` scheme.
  // The lowering computes ACC element (r, c) in the dword lane of intrinsic
  // `vpdpwssd` #(i, cb) where i = r%4, cb = c%4... — concretely the 256 i32
  // are ordered (rlo, chi, rhi, clo) with r = 4*rhi + rlo, c = 4*chi + clo.
  // As a TileSwizzle: split M into [rhi(4), rlo(4)] and N into [chi(4),
  // clo(4)] (expanded dims 0..3), then permute to (rlo, chi, rhi, clo).
  if (mma == MMAIntrinsic::MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16 &&
      operandIdx == 2) {
    TileSwizzle swizzle;
    swizzle.expandShape().resize(2);
    swizzle.expandShape()[0] = {Dim::internal(4), Dim::internal(4)};
    swizzle.expandShape()[1] = {Dim::internal(4), Dim::internal(4)};
    swizzle.permutation() = {1, 2, 0, 3};
    return swizzle;
  }

  auto maybeMnkTuple = getRowMajorTilesMNKShape(mma);
  if (!maybeMnkTuple) {
    // Whenever one adds support for a new intrinsic that doesn't have a
    // row-major tile layout, new logic goes here.
    assert(false && "Non-row-major-tile intrinsics not yet implemented.");
    return TileSwizzle();
  }
  auto [mSize, nSize, kSize] = *maybeMnkTuple;
  TileSwizzle swizzle;
  swizzle.expandShape().resize(2);
  auto expandIfNonUnit = [](TileSwizzle &swizzle, int dim, int size) {
    if (size > 1) {
      Codegen::expand(swizzle, dim, TileSwizzle::Dim::internal(size));
    }
  };

  // expandShape[0] is the outer physical dim and expandShape[1] is the inner
  // physical dim, with identity permutation. LHS is (M, K), RHS is (N, K),
  // ACC is (M, N).
  if (operandIdx == 0) {
    constexpr int M = 0, K = 1;
    expandIfNonUnit(swizzle, K, kSize);
    expandIfNonUnit(swizzle, M, mSize);
  } else if (operandIdx == 1) {
    constexpr int N = 0, K = 1;
    expandIfNonUnit(swizzle, K, kSize);
    expandIfNonUnit(swizzle, N, nSize);
  } else {
    constexpr int M = 0, N = 1;
    expandIfNonUnit(swizzle, N, nSize);
    expandIfNonUnit(swizzle, M, mSize);
  }
  return fixupSwizzle(std::move(swizzle));
}

Codegen::TileSwizzle getSwizzle(IREE::CPU::DataTiledMMAAttr mma,
                                int operandIdx) {
  using TileSwizzle = Codegen::TileSwizzle;
  TileSwizzle swizzle = getIntrinsicSwizzle(mma.getIntrinsic(), operandIdx);
  TileSwizzle::Dim intrinsicsM =
      TileSwizzle::Dim::crossIntrinsic(mma.getIntrinsicsM());
  TileSwizzle::Dim intrinsicsN =
      TileSwizzle::Dim::crossIntrinsic(mma.getIntrinsicsN());
  TileSwizzle::Dim intrinsicsK =
      TileSwizzle::Dim::crossIntrinsic(mma.getIntrinsicsK());
  // Each swizzle is built as (outer physical dim, inner physical dim) in
  // expandShape[0], expandShape[1]. LHS is (M, K), RHS is (N, K), ACC is
  // (M, N). The expansion below injects the intrinsics_* cross-intrinsic
  // factors into whichever group represents each logical dim.
  if (operandIdx == 0) {
    constexpr int M = 0, K = 1;
    if (intrinsicsK.size() > 1) {
      Codegen::expand(swizzle, K, intrinsicsK);
    }
    if (intrinsicsM.size() > 1) {
      Codegen::expand(swizzle, M, intrinsicsM);
    }
  } else if (operandIdx == 1) {
    constexpr int N = 0, K = 1;
    if (intrinsicsK.size() > 1) {
      Codegen::expand(swizzle, K, intrinsicsK);
    }
    if (intrinsicsN.size() > 1) {
      Codegen::expand(swizzle, N, intrinsicsN);
    }
  } else {
    constexpr int M = 0, N = 1;
    if (intrinsicsN.size() > 1) {
      Codegen::expand(swizzle, N, intrinsicsN);
    }
    if (intrinsicsM.size() > 1) {
      Codegen::expand(swizzle, M, intrinsicsM);
    }
  }
  return swizzle;
}

std::tuple<Type, Type, Type> getABCElementTypes(MLIRContext *ctx,
                                                MMAIntrinsic intrinsic) {
  Type f64 = Float64Type::get(ctx);
  Type f32 = Float32Type::get(ctx);
  Type f16 = Float16Type::get(ctx);
  Type bf16 = BFloat16Type::get(ctx);
  Type i32 = IntegerType::get(ctx, 32);
  Type i16 = IntegerType::get(ctx, 16);
  Type i8 = IntegerType::get(ctx, 8);
  switch (intrinsic) {
  case MMAIntrinsic::None:
    return {Type(), Type(), Type()};
  case MMAIntrinsic::MMA_X86_AVX2_FMA_1x8x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX2_FMA_8x1x1_F32_F32:
    return {f32, f32, f32};
  case MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64:
  case MMAIntrinsic::MMA_X86_AVX512_8x1x1_F64_F64:
    return {f64, f64, f64};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x1_F32_F32:
    return {f32, f32, f32};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x1_F32_F16_CASTF32:
    return {f16, f16, f32};
  case MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16:
  case MMAIntrinsic::MMA_X86_AVX512FP16_32x1x1_F16_F16:
    return {f16, f16, f16};
  case MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16:
  case MMAIntrinsic::MMA_X86_AVX512BF16_16x1x2_F32_BF16:
    return {bf16, bf16, f32};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x2_I32_I16:
    return {i16, i16, i32};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16:
    return {i8, i8, i32};
  // vpdpbusd: returned types preserve signedness for the cost model to
  // match against the encoding's `element_types`; tile types in IR are
  // signless and get stripped in `getUndistributedTileTypes` below.
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x4_I32_UI8_I8:
    return {IntegerType::get(ctx, 8, IntegerType::Unsigned), i8, i32};
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x4_I32_I8_UI8:
    return {i8, IntegerType::get(ctx, 8, IntegerType::Unsigned), i32};
  case MMAIntrinsic::MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32:
  case MMAIntrinsic::MMA_ARM_SVE_FMLA_4VLx1x1_F32_F32:
    return {f32, f32, f32};
  default:
    return {Type(), Type(), Type()};
  }
}

/// Returns the (LHS, RHS, ACC) element types for `attr`, possibly carrying
/// signedness annotations. For most `MMAIntrinsic` values these are baked
/// into the enum and the `MMAIntrinsic`-only overload above suffices. The
/// `MMA_GENERIC_SCALAR_1x1x1_REG*` family is type-polymorphic — its
/// element types live on the attr's `lhs_type` / `rhs_type` / `acc_type`
/// parameters and are returned as-is so the lowering can read the
/// signedness to pick `arith.extsi` vs `arith.extui`.
/// Vector tile types in IR (via `getUndistributedTileTypes`) are signless;
/// the signedness here is for matching against the encoding's
/// `element_types` and for routing in the lowering.
static std::tuple<Type, Type, Type>
getABCElementTypes(MLIRContext *context, IREE::CPU::DataTiledMMAAttr attr) {
  MMAIntrinsic intrinsic = attr.getIntrinsic();
  if (isGenericScalar(intrinsic)) {
    return {attr.getLhsType(), attr.getRhsType(), attr.getAccType()};
  }
  return getABCElementTypes(context, intrinsic);
}

//===----------------------------------------------------------------------===//
// DataTiledMMA Attributes
//===----------------------------------------------------------------------===//

int64_t DataTiledMMAAttr::getExpectedNumInputs() const { return 2; }

int64_t DataTiledMMAAttr::getExpectedNumOutputs() const { return 1; }

LogicalResult
DataTiledMMAAttr::verifyIndexingMaps(ArrayRef<AffineMap> maps) const {
  return linalg::inferContractionDims(maps);
}

void DataTiledMMAAttr::getUndistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  MLIRContext *ctx = getContext();
  MMAIntrinsic intrinsic = getIntrinsic();
  if (intrinsic == MMAIntrinsic::None) {
    result.clear();
    return;
  }
  auto [aType, bType, cType] = getABCElementTypes(ctx, *this);
  // Tile types in IR are signless (IREE convention); strip the signedness
  // that `getABCElementTypes` carries for cost-model matching.
  auto signless = [&](Type t) -> Type {
    if (auto intTy = dyn_cast_if_present<IntegerType>(t);
        intTy && !intTy.isSignless()) {
      return IntegerType::get(ctx, intTy.getWidth());
    }
    return t;
  };
  aType = signless(aType);
  bType = signless(bType);
  cType = signless(cType);
  result.assign({Codegen::getTileVectorType(getSwizzle(*this, 0), aType),
                 Codegen::getTileVectorType(getSwizzle(*this, 1), bType),
                 Codegen::getTileVectorType(getSwizzle(*this, 2), cType)});
}

void DataTiledMMAAttr::getDistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  // CPU has no thread distribution; use same tile types as undistributed.
  getUndistributedTileTypes(result);
}

std::optional<SmallVector<int64_t, 2>>
DataTiledMMAAttr::getUndistributedTileDimExpansion(int64_t operandIndex,
                                                   int64_t logicalDim) const {
  return std::nullopt;
}

LogicalResult DataTiledMMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, uint32_t operandIndex, Value laneId,
    ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) const {
  return failure();
}

Attribute DataTiledMMAAttr::getDistributionMappingKind() const {
  return Attribute();
}

OpFoldResult
DataTiledMMAAttr::getDistributionWorkerCount(OpBuilder &builder, Location loc,
                                             Operation *opToDistribute) const {
  return OpFoldResult();
}

// Lowers one `MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16` intrinsic: a full
// 16x16x2 i8 matmul tile, mirroring the mmt4d ukernel's hot loop.
//
// Operands (already distributed by the swizzle machinery):
//   `lhs`, `rhs`: vector<32xi8>, a row-major 16x2 i8 panel.
//   `acc`:        vector<256xi32>, the 16x16 i32 tile in the block-interleaved
//                 layout from `getIntrinsicSwizzle` — ordered (rlo, chi, rhi,
//                 clo) with r = 4*rhi + rlo, c = 4*chi + clo. So the 16 i32 at
//                 offset (4*rlo + chi)*16 are one `vpdpwssd` accumulator: its
//                 dword `4*rhi + clo` holds ACC element (r, c).
//
// Per call: one `vpmovsxbw` widen of each i8 panel to <32xi16>; 4 `vpshufd`
// fan each LHS row across its 128-bit lane; 4 128-bit-block broadcasts spread
// each group of 4 RHS columns across all lanes; 16 `vpdpwssd` over the grid.
static Value lowerX86Avx512Vnni16x16x2I8(OpBuilder &b, Location loc, Value lhs,
                                         Value rhs, Value acc) {
  Type i16 = b.getI16Type();
  Type i32 = b.getI32Type();
  auto v32i16 = VectorType::get({32}, i16);
  auto v16i32 = VectorType::get({16}, i32);

  // Widen each i8 panel to i16 once (one `vpmovsxbw`), then view as 16 i32
  // dwords so the dword shuffles below are lane-addressable. Dword `x` holds
  // the (k=0, k=1) i16 pair of LHS row `x` (resp. RHS column `x`).
  Value lhsDwords = vector::BitCastOp::create(
      b, loc, v16i32, arith::ExtSIOp::create(b, loc, v32i16, lhs));
  Value rhsDwords = vector::BitCastOp::create(
      b, loc, v16i32, arith::ExtSIOp::create(b, loc, v32i16, rhs));

  // lhsDup[i]: `vpshufd` broadcasting dword `4*lane + i` across each 128-bit
  // lane, so lane L holds LHS row 4*L + i replicated to all 4 dwords.
  // rhsBcast[cb]: broadcast the 128-bit block of columns [4*cb, 4*cb+4) to all
  // 4 lanes (a `vbroadcasti32x4`).
  SmallVector<Value, 4> lhsDup, rhsBcast;
  for (int s = 0; s < 4; ++s) {
    SmallVector<int64_t, 16> dupMask, bcastMask;
    for (int lane = 0; lane < 4; ++lane) {
      for (int e = 0; e < 4; ++e) {
        dupMask.push_back(4 * lane + s);
        bcastMask.push_back(4 * s + e);
      }
    }
    lhsDup.push_back(
        vector::ShuffleOp::create(b, loc, lhsDwords, lhsDwords, dupMask));
    rhsBcast.push_back(
        vector::ShuffleOp::create(b, loc, rhsDwords, rhsDwords, bcastMask));
  }

  // 16 `vpdpwssd` over the 4x4 grid. Accumulator #(rlo, chi) lives at flat
  // offset (4*rlo + chi)*16 in the (rlo, chi, rhi, clo)-ordered ACC vector.
  Value result = acc;
  for (int rlo = 0; rlo < 4; ++rlo) {
    for (int chi = 0; chi < 4; ++chi) {
      int64_t offset = (4 * rlo + chi) * 16;
      Value accZmm = vector::ExtractStridedSliceOp::create(
          b, loc, result, ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{16},
          ArrayRef<int64_t>{1});
      Value a16 = vector::BitCastOp::create(b, loc, v32i16, lhsDup[rlo]);
      Value b16 = vector::BitCastOp::create(b, loc, v32i16, rhsBcast[chi]);
      Value dot =
          LLVM::CallIntrinsicOp::create(
              b, loc, v16i32, b.getStringAttr("llvm.x86.avx512.vpdpwssd.512"),
              ValueRange{accZmm, a16, b16})
              .getResult(0);
      result = vector::InsertStridedSliceOp::create(
          b, loc, dot, result, ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{1});
    }
  }
  return result;
}

// Lowers a MMAIntrinsic to a llvm.call_intrinsic op, plus any necessary
// additional ops (potentially broadcasting or widening LHS/RHS or creating an
// add op if the intrinsic isn't already adding the accumulator).
static Value createCpuMmaIntrinsicCall(OpBuilder &builder, Location loc,
                                       MMAIntrinsic intrinsic, Value lhs,
                                       Value rhs, Value acc) {
  // The 16x16x2 i8 intrinsic processes whole panels and has its own widen /
  // shuffle scheme; it bypasses the per-row widen + broadcast path below.
  if (intrinsic == MMAIntrinsic::MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16) {
    return lowerX86Avx512Vnni16x16x2I8(builder, loc, lhs, rhs, acc);
  }
  // Sign-/float-extend a vector to a wider element type. Used by the
  // *_CASTF32 (f16 → f32) and *_CASTI16 (i8 → i16) variants where the
  // intrinsic only exists at the wider type.
  auto widen = [&builder, loc](Value v, Type wider) -> Value {
    auto vt = cast<VectorType>(v.getType());
    auto wideTy = VectorType::get(vt.getShape(), wider);
    if (isa<FloatType>(vt.getElementType())) {
      return arith::ExtFOp::create(builder, loc, wideTy, v);
    }
    return arith::ExtSIOp::create(builder, loc, wideTy, v);
  };

  // For *_CAST* intrinsics, widen lhs/rhs to the intrinsic's element type
  // *before* the broadcast below. The alternative — widening after the
  // broadcast, when the smaller operand has already been replicated to the
  // full lane count — would re-do the i8 → i16 (or f16 → f32) extension on
  // every M row of the unrolled tile and force LLVM to scalarize the widen
  // into a per-row `vpbroadcastw` + `vpandq` + `vpsrlvw` sequence around
  // each FMA. Widening the narrow source vector once before the broadcast
  // keeps the per-row inner loop down to just the FMA (with `m_bcst`).
  Type widenTo;
  switch (intrinsic) {
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x1_F32_F16_CASTF32:
    widenTo = builder.getF32Type();
    break;
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x2_I32_I8_CASTI16:
    widenTo = builder.getI16Type();
    break;
  default:
    break;
  }
  if (widenTo) {
    lhs = widen(lhs, widenTo);
    rhs = widen(rhs, widenTo);
  }

  // Replicate whichever of LHS/RHS has fewer lanes up to the other's lane
  // count, so both reach the intrinsic's flat lane width. In the natural
  // 1×N×K orientation that's the LHS (M=1) broadcast across the N lanes of
  // the RHS; in the M↔N-swapped N×1×K orientation it's the RHS (N=1) that
  // gets broadcast. All x86 LLVM intrinsics we target here (AVX-512 FMA,
  // VNNI vpdpwssd, BF16 dpbf16ps, integer pmaddwd) take same-width vector
  // operands — there is no single-lane-input variant.
  //
  // We emit the replication as the splat-of-a-single-packed-lane idiom:
  // bitcast the K-element source to a 1-lane vector whose lane covers the
  // full `K * elem_bits` broadcast unit, splat to `replicate` lanes via a
  // zero-mask `vector.shuffle`, and bitcast back to the flat element type.
  // This is the canonical IR shape that LLVM's x86 instruction selector
  // pattern-matches to recover the ISA-level `{1toN}` broadcast-from-memory
  // EVEX operand (`m32bcst`/`m64bcst`). The natural alternative — directly
  // shuffling `<K x elem>` to `<replicate*K x elem>` with the interleaved
  // mask `[0,...,K-1, 0,...,K-1, ...]`, or going through `vector.broadcast`
  // to a (replicate, K) 2-D shape and then `vector.shape_cast`-ing to flat
  // — is semantically equivalent but lowers to a `<K x elem>` shufflevector
  // that ISel does *not* recognize as a broadcast and so emits as a
  // separate `vbroadcastss`/`vbroadcastsd` before each FMA, doubling the
  // per-row uop count of the hot inner loop. For K=1 the bitcast pair is a
  // width-preserving no-op LLVM elides.
  // TODO(24311): Arm's by-element FMA (`fmla.4s vd, vn, vm[idx]`) is exposed
  // via separate intrinsics (e.g. `llvm.aarch64.neon.fma.lane.v4f32`) that
  // take `(vector, vector, lane_idx)`; when we add Arm support, those cases
  // should bypass this replication and emit the lane-index intrinsic directly.
  auto lhsType = cast<VectorType>(lhs.getType());
  auto rhsType = cast<VectorType>(rhs.getType());
  // Tracks whether the broadcast landed on lhs; used by the symmetric
  // intrinsic cases below to route the broadcast operand into the
  // m_bcst-foldable slot (third operand for dpbf16ps/vpdpwssd/pmaddwd, second
  // operand for FMA's `a*b+c` form).
  bool lhsIsBroadcast = false;
  if (lhsType.getNumElements() != rhsType.getNumElements()) {
    // `bcastSrc` is the operand being replicated; `bcastDst` is the operand
    // whose lane count we match. They alias the underlying lhs/rhs through
    // pointers so the result flows back into the variable used by the switch
    // below.
    Value *bcastSrc = &lhs;
    Value *bcastDst = &rhs;
    VectorType bcastSrcType = lhsType;
    VectorType bcastDstType = rhsType;
    if (bcastSrcType.getNumElements() > bcastDstType.getNumElements()) {
      std::swap(bcastSrc, bcastDst);
      std::swap(bcastSrcType, bcastDstType);
    }
    lhsIsBroadcast = (bcastSrc == &lhs);
    int64_t srcN = bcastSrcType.getNumElements();
    int64_t replicate = bcastDstType.getNumElements() / srcN;
    // The broadcast unit is `srcN` packed source elements. When it is a single
    // element, splat through that element's own type; when it packs several,
    // no scalar type names the unit, so splat through an integer of the unit's
    // bit width. x86 ISel's m_bcst fold keys on the splat *shape* (below) and
    // the broadcast-load width (`m32bcst`/`m64bcst` match any 32-/64-bit load,
    // integer or float) -- not on a float element type.
    Type srcElemTy = bcastSrcType.getElementType();
    Type laneTy = srcN == 1 ? srcElemTy
                            : Type(builder.getIntegerType(
                                  srcElemTy.getIntOrFloatBitWidth() * srcN));
    auto singleLaneTy = VectorType::get({1}, laneTy);
    auto replicatedTy = VectorType::get({replicate}, laneTy);
    Value asSingleLane =
        vector::BitCastOp::create(builder, loc, singleLaneTy, *bcastSrc);
    // Extract scalar + `vector.broadcast` so this lowers to LLVM's canonical
    // `insertelement <N x T> poison, T, 0` + `shufflevector <N x T> ...,
    // <N x i32> zeroinitializer` splat shape (the same shape the ukernel's
    // `_mm512_set1_ps` produces, and the one x86 ISel folds into m_bcst).
    // A direct `shufflevector <1 x T> -> <N x T>` is semantically equivalent
    // but does not pattern-match.
    Value scalar = vector::ExtractOp::create(builder, loc, asSingleLane,
                                             ArrayRef<int64_t>{0});
    Value splatted =
        vector::BroadcastOp::create(builder, loc, replicatedTy, scalar);
    *bcastSrc = vector::BitCastOp::create(builder, loc, bcastDstType, splatted);
  }

  // Emit llvm.call_intrinsic.
  auto call = [&builder, loc](StringRef name, Type resultType,
                              ValueRange args) -> Value {
    return LLVM::CallIntrinsicOp::create(builder, loc, resultType,
                                         builder.getStringAttr(name), args)
        .getResult(0);
  };

  Type accType = acc.getType();
  // Most x86 MMAs supported here are LHS/RHS-symmetric (FMA, dpbf16ps,
  // vpdpwssd, pmaddw): natural and swapped orientations share the same LLVM
  // intrinsic and arg order. For these we route the broadcasted operand into
  // the m_bcst-foldable slot: the third operand for dpbf16ps/vpdpwssd/
  // pmaddwd, and the `b` operand (= second mul) for FMA's `a*b+c`. The
  // exception is vpdpbusd, which is asymmetric (first byte source unsigned,
  // second signed): we keep its existing sign-aware routing — when the
  // broadcast happens to land in the signed slot, that maps to vpdpbusd's
  // m_bcst-foldable slot too; when it lands on the unsigned side, no fold is
  // available (the ISA's m_bcst is on the signed operand).
  //
  // For *_CAST* variants (f16 → f32, i8 → i16), the widening already ran at
  // the top of this function, so lhs/rhs reach the intrinsic call at the
  // wider type without any per-row widen in the unrolled tile.
  Value bcst = lhsIsBroadcast ? lhs : rhs;
  Value full = lhsIsBroadcast ? rhs : lhs;
  switch (intrinsic) {
  case MMAIntrinsic::MMA_X86_AVX2_FMA_1x8x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX2_FMA_8x1x1_F32_F32:
    return call("llvm.fma.v8f32", accType, ValueRange{full, bcst, acc});
  case MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64:
  case MMAIntrinsic::MMA_X86_AVX512_8x1x1_F64_F64:
    return call("llvm.fma.v8f64", accType, ValueRange{full, bcst, acc});
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x1_F32_F16_CASTF32:
    return call("llvm.fma.v16f32", accType, ValueRange{full, bcst, acc});
  case MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16:
  case MMAIntrinsic::MMA_X86_AVX512FP16_32x1x1_F16_F16:
    return call("llvm.fma.v32f16", accType, ValueRange{full, bcst, acc});
  case MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16:
  case MMAIntrinsic::MMA_X86_AVX512BF16_16x1x2_F32_BF16:
    return call("llvm.x86.avx512bf16.dpbf16ps.512", accType,
                ValueRange{acc, full, bcst});
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x2_I32_I8_CASTI16:
    return call("llvm.x86.avx512.vpdpwssd.512", accType,
                ValueRange{acc, full, bcst});
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512_16x1x2_I32_I8_CASTI16: {
    return arith::AddIOp::create(
        builder, loc, acc,
        call("llvm.x86.avx512.pmaddw.d.512", accType, ValueRange{full, bcst}));
  }
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x4_I32_UI8_I8:
    // LHS unsigned, RHS signed — vpdpbusd takes (acc, unsigned, signed).
    return call("llvm.x86.avx512.vpdpbusd.512", accType,
                ValueRange{acc, lhs, rhs});
  case MMAIntrinsic::MMA_X86_AVX512VNNI_16x1x4_I32_I8_UI8:
    // LHS signed, RHS unsigned — swap to put the unsigned operand first.
    return call("llvm.x86.avx512.vpdpbusd.512", accType,
                ValueRange{acc, rhs, lhs});
  default:
    return {};
  }
}

/// Lowers a `DataTiledMMAAttr` whose intrinsic is one of the
/// `MMA_GENERIC_SCALAR_1x1x1_REG*` cases directly to a single
/// `vector.contract`. Since the intrinsic's tile shape is 1×1×1, the only
/// non-unit dims in each operand tile come from `intrinsics_{m,n,k}`, and
/// the `Codegen::expand`/`fixupSwizzle` machinery places them in row-major
/// (M, K) / (N, K) / (M, N) order. Collapse to that rank-2 form with
/// `vector.shape_cast` regardless of where the non-unit dim(s) sit in the
/// operand tile, then `shape_cast` the contract result back to the ACC
/// tile's original shape so callers downstream see the expected type.
static LogicalResult lowerGenericScalarToVectorContract(
    OpBuilder &builder, Location loc, IREE::CPU::DataTiledMMAAttr attr,
    ValueRange inputs, ValueRange outputs, SmallVectorImpl<Value> &results) {
  assert(isGenericScalar(attr.getIntrinsic()) &&
         "lowerGenericScalarToVectorContract only handles "
         "MMA_GENERIC_SCALAR_1x1x1_REG* intrinsics");
  int64_t M = attr.getIntrinsicsM();
  int64_t N = attr.getIntrinsicsN();
  int64_t K = attr.getIntrinsicsK();
  auto reshape = [&](Value v, ArrayRef<int64_t> targetShape) -> Value {
    auto vt = cast<VectorType>(v.getType());
    auto targetTy = VectorType::get(targetShape, vt.getElementType());
    if (vt == targetTy) {
      return v;
    }
    return vector::ShapeCastOp::create(builder, loc, targetTy, v);
  };
  Value lhs = reshape(inputs[0], {M, K});
  Value rhs = reshape(inputs[1], {N, K});
  auto accTy = cast<VectorType>(outputs[0].getType());
  Value acc = reshape(outputs[0], {M, N});
  Type accElem = accTy.getElementType();
  // For the generic intrinsic, ACC is always at least as wide as LHS/RHS,
  // and they're either all float or all integer (the cost model only picks
  // this intrinsic when element types are mutually consistent that way).
  // Signedness lives on the attr's `lhs_type` / `rhs_type` (the operand
  // vector types are signless storage) and picks ExtSI vs. ExtUI; ACC is
  // treated as signed.
  auto isUnsigned = [](Type t) {
    auto intTy = dyn_cast_if_present<IntegerType>(t);
    return intTy && intTy.isUnsigned();
  };
  auto widenToAcc = [&](Value v, bool unsignedSrc) -> Value {
    auto vt = cast<VectorType>(v.getType());
    if (vt.getElementType() == accElem) {
      return v;
    }
    auto wideTy = VectorType::get(vt.getShape(), accElem);
    if (isa<FloatType>(accElem)) {
      return arith::ExtFOp::create(builder, loc, wideTy, v);
    }
    return unsignedSrc ? Value(arith::ExtUIOp::create(builder, loc, wideTy, v))
                       : Value(arith::ExtSIOp::create(builder, loc, wideTy, v));
  };
  lhs = widenToAcc(lhs, isUnsigned(attr.getLhsType()));
  rhs = widenToAcc(rhs, isUnsigned(attr.getRhsType()));
  AffineExpr m = builder.getAffineDimExpr(0);
  AffineExpr n = builder.getAffineDimExpr(1);
  AffineExpr k = builder.getAffineDimExpr(2);
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  // LHS is (M, K), RHS is (N, K), ACC is (M, N) — same as the
  // `iree_codegen.inner_tiled` op's own indexing maps for CPU.
  vector::IteratorType par = vector::IteratorType::parallel;
  vector::IteratorType red = vector::IteratorType::reduction;
  Value contractResult = vector::ContractionOp::create(
      builder, loc, lhs, rhs, acc, MapList{{m, k}, {n, k}, {m, n}},
      ArrayRef<vector::IteratorType>{par, par, red});
  results.push_back(reshape(contractResult, accTy.getShape()));
  return success();
}

LogicalResult DataTiledMMAAttr::buildUnderlyingOperations(
    OpBuilder &builder, Location loc, ValueRange inputs, ValueRange outputs,
    SmallVectorImpl<Value> &results) const {
  SmallVector<VectorType> regTypes;
  getDistributedTileTypes(regTypes);
  if (inputs.size() != 2 || outputs.size() != 1 ||
      !llvm::equal(regTypes,
                   llvm::concat<Type>(inputs.getTypes(), outputs.getTypes()))) {
    return failure();
  }

  MMAIntrinsic intrinsic = getIntrinsic();
  // The type-polymorphic generic intrinsic is row-major (1×1×1 base) and
  // bypasses the swizzle/distribution machinery: it lowers directly to a
  // single `vector.contract` over the unrolled operand tiles, the way
  // `linalg.mmt4d` would.
  if (isGenericScalar(intrinsic)) {
    return lowerGenericScalarToVectorContract(builder, loc, *this, inputs,
                                              outputs, results);
  }
  auto emitIntrinsic = [&](OpBuilder &b, Location loc, Value lhs, Value rhs,
                           Value acc) -> Value {
    return createCpuMmaIntrinsicCall(b, loc, intrinsic, lhs, rhs, acc);
  };
  return Codegen::buildDataTiledMMAUnderlyingOperations(
      builder, loc, getSwizzle(*this, /*operandIdx=*/0),
      getSwizzle(*this, /*operandIdx=*/1), getSwizzle(*this, /*operandIdx=*/2),
      getIntrinsicsM(), getIntrinsicsN(), getIntrinsicsK(), inputs, outputs,
      emitIntrinsic, results);
}

//===----------------------------------------------------------------------===//
// InnerTiledSemanticsAttr
//===----------------------------------------------------------------------===//

void InnerTiledSemanticsAttr::getTileTypes(
    IREE::Codegen::InnerTileDescAttrInterface kind,
    SmallVectorImpl<VectorType> &result) const {
  // CPU has no thread distribution; always use undistributed tile types.
  kind.getUndistributedTileTypes(result);
}

bool InnerTiledSemanticsAttr::getOpaque() const { return false; }

//===----------------------------------------------------------------------===//
// UKernelProviderAttr
//===----------------------------------------------------------------------===//

constexpr StringLiteral kHalExecutableObjectsAttrName =
    "hal.executable.objects";

// Walks parents from `op` to return the nearest `hal.executable.objects`
// array attribute. If a parent `hal.executable.variant` is reached, its
// `objects` attribute is returned. Adapted from
// `ExecutableTargetAttr::lookup`. This is how user-supplied bitcode reaches
// the ukernel provider: a source MLIR program can attach
// `hal.executable.objects` at any ancestor of the op (commonly the
// module, the executable variant, or the op itself).
static ArrayAttr lookUpExecutableObjects(Operation *op) {
  MLIRContext *context = op->getContext();
  auto attrId = StringAttr::get(context, kHalExecutableObjectsAttrName);
  while (op) {
    if (auto variantOp = dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
      if (std::optional<ArrayAttr> objects = variantOp.getObjects()) {
        return *objects;
      }
    }
    if (auto attr = op->getAttrOfType<ArrayAttr>(attrId)) {
      return attr;
    }
    op = op->getParentOp();
  }
  return {};
}

// Returns true if `path` matches `<ukernelName>.<features>.bc`, the
// filename convention used by the bitcode build rules under
// `compiler/plugins/target/LLVMCPU/builtins/ukernel/`. The `<features>`
// part is opaque to the lookup; this CPU framework currently picks
// whichever built-in (or user-supplied) variant exists for the given
// algorithm name. Multi-feature-variant disambiguation will arrive with
// the SelectUKernels pass.
static bool isMatchingUKernelFile(StringRef path, StringRef ukernelName) {
  if (!path.ends_with(".bc")) {
    return false;
  }
  if (!path.starts_with(ukernelName)) {
    return false;
  }
  // Either exact name (`<name>.bc`) or features-suffixed
  // (`<name>.<features>.bc`).
  size_t after = ukernelName.size();
  return after == path.size() - 3 ||
         (after < path.size() - 3 && path[after] == '.');
}

// Returns an `ExecutableObjectAttr` carrying the bitcode for `ukernelName`.
// First searches `sourceExecutableObjects` (the user-supplied
// `hal.executable.objects` ancestor — the bring-your-own-ukernel path),
// then falls back to the global `EmbeddedDataDirectory` (populated at
// LLVMCPU plugin init from the embedded TOC). Returns null if no match.
static IREE::HAL::ExecutableObjectAttr
getUKernelBitcode(MLIRContext *context, ArrayAttr sourceExecutableObjects,
                  StringRef ukernelName) {
  if (sourceExecutableObjects) {
    for (Attribute a : sourceExecutableObjects) {
      auto object = dyn_cast<IREE::HAL::ExecutableObjectAttr>(a);
      if (object && isMatchingUKernelFile(object.getPath(), ukernelName)) {
        return object;
      }
    }
  }
  std::optional<StringRef> matchedFilename;
  std::optional<StringRef> matchedBytes;
  EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
    for (auto &entry : dir.getMap()) {
      if (isMatchingUKernelFile(entry.getKey(), ukernelName)) {
        matchedFilename = entry.getKey();
        matchedBytes = entry.getValue();
        return;
      }
    }
  });
  if (!matchedFilename) {
    return {};
  }
  AsmResourceBlob blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(
      ArrayRef<char>(matchedBytes->data(), matchedBytes->size()));
  auto bitcodeDenseAttr = DenseI8ResourceElementsAttr::get(
      VectorType::get({static_cast<int64_t>(matchedBytes->size())},
                      IntegerType::get(context, 8)),
      *matchedFilename, std::move(blob));
  return IREE::HAL::ExecutableObjectAttr::get(
      context, StringAttr::get(context, *matchedFilename),
      cast<IREE::Util::SerializableAttrInterface>(bitcodeDenseAttr));
}

// Idempotently appends `bitcodeObject` to the `hal.executable.objects`
// array on `op`. Returns true if the array already contained an
// equivalent entry (i.e. nothing was changed).
static bool addBitcodeObjectIfMissing(Operation *op, StringAttr attrId,
                                      Attribute bitcodeObject) {
  ArrayAttr existing = op->getAttrOfType<ArrayAttr>(attrId);
  if (existing) {
    for (Attribute a : existing) {
      if (a == bitcodeObject) {
        return true;
      }
    }
  }
  SmallVector<Attribute> objects;
  if (existing) {
    objects.append(existing.begin(), existing.end());
  }
  objects.push_back(bitcodeObject);
  op->setAttr(attrId, ArrayAttr::get(op->getContext(), objects));
  return false;
}

bool attachUKernelBitcodeOnOp(Operation *op, StringRef name) {
  MLIRContext *context = op->getContext();
  ArrayAttr sourceExecutableObjects = lookUpExecutableObjects(op);
  IREE::HAL::ExecutableObjectAttr bitcodeObject =
      getUKernelBitcode(context, sourceExecutableObjects, name);
  if (!bitcodeObject) {
    return false;
  }
  auto attrId = StringAttr::get(context, kHalExecutableObjectsAttrName);
  addBitcodeObjectIfMissing(op, attrId, bitcodeObject);

  // Also attach to the enclosing `hal.executable.variant`'s `objects` (a
  // first-class operand attribute, not a discardable attr). This is the
  // long-lived home for the bitcode: it survives codegen passes that
  // would otherwise strip discardable attributes off intermediate ops,
  // and matches the destination that `iree-hal-hoist-executable-objects`
  // would eventually hoist to. Doing it eagerly here just shortens the
  // distance the bitcode has to travel.
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto variantOp = dyn_cast<IREE::HAL::ExecutableVariantOp>(parent)) {
      SmallVector<Attribute> variantObjects;
      if (std::optional<ArrayAttr> existing = variantOp.getObjects()) {
        for (Attribute a : *existing) {
          if (a == bitcodeObject) {
            return true;
          }
          variantObjects.push_back(a);
        }
      }
      variantObjects.push_back(bitcodeObject);
      variantOp.setObjectsAttr(ArrayAttr::get(context, variantObjects));
      return true;
    }
    parent = parent->getParentOp();
  }
  return true;
}

// Returns the index, in the ACC operand's shape, of the innermost
// CrossIntrinsic dimension (the N cross-intrinsic dim for our layouts), or
// nullopt if it is dynamic / absent. The ukernel needs this dim's stride to
// address each unrolled intrinsic's ACC fragment.
//
// Example: with `MMA_X86_AVX512BF16_1x16x2` (each intrinsic produces a 1x16
// f32 fragment) and intrinsics_m = intrinsics_n = 2, the ACC tile holds a 2x2
// grid of such fragments. The two CrossIntrinsic dims are that grid's M and N;
// the innermost is N, so this returns the index of the N-grid dim in the ACC
// result shape, whose stride is the element distance from fragment (m, n) to
// (m, n+1).
static std::optional<unsigned>
getAccInnermostCrossIntrinsicDim(IREE::Codegen::InnerTiledOp op,
                                 DataTiledMMAAttr mma) {
  auto outputType = dyn_cast<ShapedType>(op.getResultTypes()[0]);
  if (!outputType) {
    return std::nullopt;
  }
  Codegen::TileSwizzle accSwizzle = getSwizzle(mma, /*operandIdx=*/2);
  SmallVector<Codegen::TileSwizzle::Dim> swizzleDims;
  for (const Codegen::TileSwizzle::ExpandShapeDimVectorType &group :
       accSwizzle.expandShape()) {
    swizzleDims.append(group.begin(), group.end());
  }
  applyPermutationToVector(swizzleDims, accSwizzle.permutation());
  int rankDiff = outputType.getRank() - static_cast<int>(swizzleDims.size());
  auto crossIntrinsic = Codegen::TileSwizzle::Dim::Kind::CrossIntrinsic;
  for (size_t i = swizzleDims.size(); i-- > 0;) {
    if (swizzleDims[i].kind() != crossIntrinsic) {
      continue;
    }
    int outputIdx = i + rankDiff;
    if (outputType.isDynamicDim(outputIdx)) {
      return std::nullopt;
    }
    return outputIdx;
  }
  // No CrossIntrinsic dims (intrinsics_m == intrinsics_n == 1): the single
  // fragment sits at the start of the inner tile.
  if (!swizzleDims.empty()) {
    return rankDiff;
  }
  return std::nullopt;
}

// Rewrites an `inner_tiled` carrying a CPU `DataTiledMMAAttr` to a
// `ukernel.generic`, threading the data-tiled-MMA scalar parameters as
// operands so the ukernel can loop over arbitrary `intrinsics_{m,n,k}`:
//   ins(lhs, rhs) outs(acc) (k_outer, intrinsics_m, intrinsics_n, intrinsics_k)
// plus a `strided_dims` entry giving the ACC's innermost cross-intrinsic
// stride.
static LogicalResult
handleInnerTiledMmaUkernel(RewriterBase &rewriter, StringRef name,
                           IREE::Codegen::InnerTiledOp op, DataTiledMMAAttr mma,
                           ArrayRef<Value> inputs, ArrayRef<Value> outputs,
                           DictionaryAttr fnDefAttrs) {
  std::optional<unsigned> accInnerDim =
      getAccInnermostCrossIntrinsicDim(op, mma);
  if (!accInnerDim) {
    return rewriter.notifyMatchFailure(
        op, "ACC innermost cross-intrinsic dim is dynamic or absent");
  }
  Location loc = op.getLoc();
  Type i32 = rewriter.getI32Type();
  auto constI32 = [&](int64_t v) {
    return arith::ConstantIntOp::create(rewriter, loc, i32, v);
  };
  // Outer-K tile count: the LHS operand's outer dims are (m_outer, k_outer),
  // so dim 1 is the K-tile count the ukernel loops over.
  Value kOuter = arith::IndexCastOp::create(
      rewriter, loc, i32,
      tensor::DimOp::create(rewriter, loc, op.getInputs()[0], 1));
  // `strided_dims` is the `ukernel.generic` ABI knob for which dims' strides
  // are passed to the C function: a list per shaped operand (here LHS, RHS,
  // ACC), each naming the dims whose stride follows that operand's
  // `(base, offset)` in the call. An empty list passes `base, offset` only; a
  // null attribute (the default, used by simpler ukernels) passes all strides.
  // Here only ACC needs one — the innermost cross-intrinsic (N) dim — so the
  // ukernel can address each unrolled intrinsic's ACC fragment. LHS/RHS are
  // contiguous in the data-tiled layout, so they take no stride argument.
  SmallVector<SmallVector<int64_t>> stridedDims(3, {});
  stridedDims[2].push_back(*accInnerDim);
  DictionaryAttr discardableAttrs = op->getDiscardableAttrDictionary();
  auto newOp = rewriter.replaceOpWithNewOp<IREE::Codegen::UKernelGenericOp>(
      op, op.getOutputs().getTypes(), name, inputs, outputs,
      ValueRange{kOuter, constI32(mma.getIntrinsicsM()),
                 constI32(mma.getIntrinsicsN()),
                 constI32(mma.getIntrinsicsK())},
      fnDefAttrs, stridedDims);
  newOp->setDiscardableAttrs(discardableAttrs);
  return success();
}

std::optional<LogicalResult> UKernelProviderAttr::createAndReplaceWithUkernelOp(
    RewriterBase &rewriter, StringRef name, DictionaryAttr targetConfiguration,
    Operation *contextualOp, ArrayRef<Value> inputs, ArrayRef<Value> outputs,
    SmallVectorImpl<Value> &otherOperands) const {
  // Idempotent: in the normal flow `LLVMCPUSelectUKernels` attached the
  // bitcode at kernel-config time, so this is a no-op. Still defensive
  // for tests / future paths that bypass SelectUKernels.
  rewriter.modifyOpInPlace(
      contextualOp, [&] { attachUKernelBitcodeOnOp(contextualOp, name); });

  // We build the `ukernel.generic` ourselves (rather than returning nullopt
  // and letting the default fallback in `LowerBitcodeUKernelsPass` handle it)
  // for two reasons: (1) to set `fn_def_attrs = {hal.import.bitcode = true}`,
  // and (2) for `inner_tiled` ops, to thread the `DataTiledMMAAttr` scalar
  // parameters as operands. The `hal.import.bitcode` flag propagates onto the
  // `func.func` declaration that `LowerUKernelOpsToCalls` synthesizes;
  // `RewriteExternCallOpToDynamicImportCallOp` (in LLVMCPU's ConvertToLLVM)
  // keys on it to *skip* the import-table indirection it otherwise applies to
  // every external call, letting the call resolve directly against the linked
  // bitcode at LLVM optimization time. Without it the new framework would
  // accidentally re-use the legacy runtime-resolved-import path.
  MLIRContext *context = rewriter.getContext();
  auto fnDefAttrs = DictionaryAttr::get(
      context, {{rewriter.getStringAttr("hal.import.bitcode"),
                 rewriter.getBoolAttr(true)}});

  // `inner_tiled` with a CPU `DataTiledMMAAttr`: thread the unrolling factors,
  // outer-K count and ACC stride so the ukernel can loop over the intrinsics.
  if (auto innerTiled = dyn_cast<IREE::Codegen::InnerTiledOp>(contextualOp)) {
    if (auto mma = dyn_cast<DataTiledMMAAttr>(innerTiled.getKind())) {
      return handleInnerTiledMmaUkernel(rewriter, name, innerTiled, mma, inputs,
                                        outputs, fnDefAttrs);
    }
  }

  // Any other op: a plain `ukernel.generic` with no extra operands.
  DictionaryAttr discardableAttrs =
      contextualOp->getDiscardableAttrDictionary();
  auto newOp = rewriter.replaceOpWithNewOp<IREE::Codegen::UKernelGenericOp>(
      contextualOp, contextualOp->getResults().getTypes(), name, inputs,
      outputs, otherOperands, fnDefAttrs,
      /*num_strided_outer_dims=*/0);
  newOp->setDiscardableAttrs(discardableAttrs);
  return success();
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREECPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::CPU
