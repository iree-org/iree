// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/MMAUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
static std::optional<std::tuple<int64_t, int64_t, int64_t>>
getRowMajorTilesMNKShape(MMAIntrinsic intrinsic) {
  using Tuple = std::tuple<int64_t, int64_t, int64_t>;
  switch (intrinsic) {
  case MMAIntrinsic::None:
    return Tuple{0, 0, 0};
  case MMAIntrinsic::MMA_X86_AVX2_FMA_1x8x1_F32_F32:
    return Tuple{1, 8, 1};
  case MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64:
    return Tuple{1, 8, 1};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
    return Tuple{1, 16, 1};
  case MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16:
    return Tuple{1, 32, 1};
  case MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
    return Tuple{1, 16, 2};
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
                                         bool transposed, int operandIdx) {
  using TileSwizzle = Codegen::TileSwizzle;
  using Dim = TileSwizzle::Dim;

  // Just one scalable intrinsic for now, to allow writing some tests.
  if (mma == MMAIntrinsic::MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32) {
    TileSwizzle swizzle;
    swizzle.expandShape().resize(2);
    if (operandIdx != 0) {
      Codegen::expand(
          swizzle, /*srcDim=*/0,
          Dim::internal(4, Dim::SymbolicMultiplier::ArmSveVLIn128bitUnits));
    }
    return fixupSwizzle(std::move(swizzle));
  }

  auto maybeMnkTuple = getRowMajorTilesMNKShape(mma);
  if (!maybeMnkTuple) {
    // Whenever one adds support for a new intrinsic that doesn't have a
    // row-major tile layout, new logic goes here.
    assert(false && "Non-row-major-tile intrinsics not yet implemented.");
    return TileSwizzle();
  }
  auto [mSize, nSize, kSize] = *maybeMnkTuple;
  // In the transposed orientation, the intrinsic's hardware (M, N) roles are
  // logically swapped: what the matmul code sees as M is driven by the
  // intrinsic's N-dim, and vice versa. Swap the sizes up front so that the
  // per-operand expansion code below can stay oblivious to orientation.
  if (transposed) {
    std::swap(mSize, nSize);
  }
  TileSwizzle swizzle;
  swizzle.expandShape().resize(2);
  auto expandIfNonUnit = [](TileSwizzle &swizzle, int dim, int size) {
    if (size > 1) {
      Codegen::expand(swizzle, dim, TileSwizzle::Dim::internal(size));
    }
  };

  // For every operand, expandShape[0] is the outer physical dim and
  // expandShape[1] is the inner physical dim, with identity permutation. For
  // the ACC in particular, `transposed_intrinsic` flips the logical (M, N) to
  // physical (N, M), which we encode by swapping which logical dim fills each
  // expandShape group rather than by a non-identity permutation. That way all
  // three operand swizzles can be read on equal footing, just like LHS (M, K)
  // and RHS (N, K).
  if (operandIdx == 0) {
    constexpr int M = 0, K = 1;
    expandIfNonUnit(swizzle, K, kSize);
    expandIfNonUnit(swizzle, M, mSize);
  } else if (operandIdx == 1) {
    constexpr int N = 0, K = 1;
    expandIfNonUnit(swizzle, K, kSize);
    expandIfNonUnit(swizzle, N, nSize);
  } else {
    int64_t accOuter = transposed ? nSize : mSize;
    int64_t accInner = transposed ? mSize : nSize;
    expandIfNonUnit(swizzle, 1, accInner);
    expandIfNonUnit(swizzle, 0, accOuter);
  }
  return fixupSwizzle(std::move(swizzle));
}

Codegen::TileSwizzle getSwizzle(IREE::CPU::DataTiledMMAAttr mma,
                                int operandIdx) {
  using TileSwizzle = Codegen::TileSwizzle;
  TileSwizzle swizzle = getIntrinsicSwizzle(
      mma.getIntrinsic(), mma.getTransposedIntrinsic(), operandIdx);
  TileSwizzle::Dim intrinsicsM =
      TileSwizzle::Dim::crossIntrinsic(mma.getIntrinsicsM());
  TileSwizzle::Dim intrinsicsN =
      TileSwizzle::Dim::crossIntrinsic(mma.getIntrinsicsN());
  TileSwizzle::Dim intrinsicsK =
      TileSwizzle::Dim::crossIntrinsic(mma.getIntrinsicsK());
  // Each swizzle is built as (outer physical dim, inner physical dim) in
  // expandShape[0], expandShape[1]. LHS is (M, K), RHS is (N, K), ACC is
  // (M, N) normally and (N, M) when `transposed_intrinsic` is set. The
  // expansion below injects the intrinsics_* cross-intrinsic factors into
  // whichever group represents each logical dim.
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
    bool transposed = mma.getTransposedIntrinsic();
    TileSwizzle::Dim accOuterIntr = transposed ? intrinsicsN : intrinsicsM;
    TileSwizzle::Dim accInnerIntr = transposed ? intrinsicsM : intrinsicsN;
    if (accInnerIntr.size() > 1) {
      Codegen::expand(swizzle, /*srcIdx=*/1, accInnerIntr);
    }
    if (accOuterIntr.size() > 1) {
      Codegen::expand(swizzle, /*srcIdx=*/0, accOuterIntr);
    }
  }
  return swizzle;
}

static std::tuple<Type, Type, Type> getABCElementTypes(MLIRContext *context,
                                                       MMAIntrinsic intrinsic) {
  Type f64 = Float64Type::get(context);
  Type f32 = Float32Type::get(context);
  Type f16 = Float16Type::get(context);
  Type bf16 = BFloat16Type::get(context);
  Type i32 = IntegerType::get(context, 32);
  Type i16 = IntegerType::get(context, 16);
  Type i8 = IntegerType::get(context, 8);
  switch (intrinsic) {
  case MMAIntrinsic::None:
    return {Type(), Type(), Type()};
  case MMAIntrinsic::MMA_X86_AVX2_FMA_1x8x1_F32_F32:
    return {f32, f32, f32};
  case MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64:
    return {f64, f64, f64};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32:
    return {f32, f32, f32};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
    return {f16, f16, f32};
  case MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16:
    return {f16, f16, f16};
  case MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16:
    return {bf16, bf16, f32};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16:
    return {i16, i16, i32};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
    return {i8, i8, i32};
  case MMAIntrinsic::MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32:
    return {f32, f32, f32};
  default:
    return {Type(), Type(), Type()};
  }
}

/// Returns the (LHS, RHS, ACC) *storage* element types for `attr` — what
/// `getDistributedTileTypes` should plumb into vector types and what the
/// inner_tiled op's operand types must agree with. For most `MMAIntrinsic`
/// values these are baked into the enum and the `MMAIntrinsic`-only
/// overload above suffices. The `MMA_GENERIC_SCALAR_1x1x1_REG*` family is
/// type-polymorphic — its element types live on the attr's `lhs_type` /
/// `rhs_type` / `acc_type` parameters. We strip integer signedness here:
/// storage is always signless, the attr keeps the `siN` / `uiN` annotation
/// for the lowering to pick `arith.extsi` vs `arith.extui`.
static std::tuple<Type, Type, Type>
getABCElementTypes(MLIRContext *context, IREE::CPU::DataTiledMMAAttr attr) {
  MMAIntrinsic intrinsic = attr.getIntrinsic();
  if (isGenericScalar(intrinsic)) {
    auto signless = [&](Type t) -> Type {
      if (auto intTy = dyn_cast_if_present<IntegerType>(t);
          intTy && !intTy.isSignless()) {
        return IntegerType::get(context, intTy.getWidth());
      }
      return t;
    };
    return {signless(attr.getLhsType()), signless(attr.getRhsType()),
            signless(attr.getAccType())};
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

/// Returns a pair where the first element is the element count of the group and
/// the second element is whether the group contains a scalable dimension.
static std::pair<int64_t, bool>
getVectorAxisSizeAndScalability(ArrayRef<Codegen::TileSwizzle::Dim> group) {
  using Dim = Codegen::TileSwizzle::Dim;
  int64_t size = 1;
  bool scalable = false;
  for (const Dim &d : group) {
    size *= d.size();
    scalable |= d.kind() == Dim::Kind::Internal &&
                d.symbolicMultiplier() != Dim::SymbolicMultiplier::One;
  }
  return {size, scalable};
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
  // Each operand's swizzle encodes its tile shape as (outer physical dim,
  // inner physical dim) in expandShape[0], expandShape[1]. This mirrors GPU's
  // DataTiledMMA, where the tile types encode the layout directly and no
  // separate `permutations` attribute is needed on `inner_tiled`. LHS is
  // (M, K), RHS is (N, K), and ACC is (M, N) for non-transposed intrinsics
  // and (N, M) for transposed ones; that transposition is baked into the ACC
  // swizzle itself (see getIntrinsicSwizzle), so we can query all three
  // operands uniformly here.
  auto tileType = [&](Codegen::TileSwizzle swizzle, Type elemType) {
    auto [outer, outerScalable] =
        getVectorAxisSizeAndScalability(swizzle.expandShape()[0]);
    auto [inner, innerScalable] =
        getVectorAxisSizeAndScalability(swizzle.expandShape()[1]);
    bool scalable[] = {outerScalable, innerScalable};
    return VectorType::get({outer, inner}, elemType, scalable);
  };
  result.assign({tileType(getSwizzle(*this, 0), aType),
                 tileType(getSwizzle(*this, 1), bType),
                 tileType(getSwizzle(*this, 2), cType)});
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

// Lowers a MMAIntrinsic to a llvm.call_intrinsic op, plus any necessary
// additional ops (potentially broadcasting or widening LHS/RHS or creating an
// add op if the intrinsic isn't already adding the accumulator).
//
// Currently assumes that if one of LHS or RHS needs to be broadcasted, then it
// musst be LHS. This corresponds to the fact that transposed intrinsics are not
// currently handled by the caller, which is just a temporary limitation.
static Value createCpuMmaIntrinsicCall(OpBuilder &builder, Location loc,
                                       MMAIntrinsic intrinsic, Value lhs,
                                       Value rhs, Value acc) {
  // Replicate LHS (M=1, K elements) across RHS's N lanes so both have the
  // intrinsic's flat lane width. Going through a (N, K) 2-D form keeps the
  // K-pair contiguous; a final shape_cast collapses to the flat 1-D vector
  // the LLVM intrinsic expects. All x86 LLVM intrinsics we target here
  // (AVX-512 FMA, VNNI vpdpwssd, BF16 dpbf16ps, integer pmaddwd) take same-
  // width vector operands — there is no narrow-input variant. The ISA-level
  // `{1toN}` broadcast-from-memory form is recovered later by the x86
  // backend's instruction selector pattern-matching this `vector.broadcast`-
  // of-load into the EVEX broadcast operand, so the explicit broadcast here
  // is what *enables* that, not a perf liability.
  // TODO(24311): Arm's by-element FMA (`fmla.4s vd, vn, vm[idx]`) is exposed
  // via separate intrinsics (e.g. `llvm.aarch64.neon.fma.lane.v4f32`) that
  // take `(vector, vector, lane_idx)`; when we add Arm support, those cases
  // should bypass this broadcast and emit the lane-index intrinsic directly.
  auto lhsType = cast<VectorType>(lhs.getType());
  auto rhsType = cast<VectorType>(rhs.getType());
  if (lhsType.getNumElements() != rhsType.getNumElements()) {
    int64_t replicate = rhsType.getNumElements() / lhsType.getNumElements();
    auto bcastType = VectorType::get({replicate, lhsType.getNumElements()},
                                     lhsType.getElementType());
    Value bcast = vector::BroadcastOp::create(builder, loc, bcastType, lhs);
    lhs = vector::ShapeCastOp::create(builder, loc, rhsType, bcast);
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

  // Emit llvm.call_intrinsic.
  auto call = [&builder, loc](StringRef name, Type resultType,
                              ValueRange args) -> Value {
    return LLVM::CallIntrinsicOp::create(builder, loc, resultType,
                                         builder.getStringAttr(name), args)
        .getResult(0);
  };

  Type f32 = builder.getF32Type();
  Type i16 = builder.getI16Type();
  Type accType = acc.getType();
  switch (intrinsic) {
  case MMAIntrinsic::MMA_X86_AVX2_FMA_1x8x1_F32_F32:
    return call("llvm.fma.v8f32", accType, ValueRange{lhs, rhs, acc});
  case MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64:
    return call("llvm.fma.v8f64", accType, ValueRange{lhs, rhs, acc});
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32:
    return call("llvm.fma.v16f32", accType, ValueRange{lhs, rhs, acc});
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
    return call("llvm.fma.v16f32", accType,
                ValueRange{widen(lhs, f32), widen(rhs, f32), acc});
  case MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16:
    return call("llvm.fma.v32f16", accType, ValueRange{lhs, rhs, acc});
  case MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16:
    return call("llvm.x86.avx512bf16.dpbf16ps.512", accType,
                ValueRange{acc, lhs, rhs});
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16:
    return call("llvm.x86.avx512.vpdpwssd.512", accType,
                ValueRange{acc, lhs, rhs});
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
    return call("llvm.x86.avx512.vpdpwssd.512", accType,
                ValueRange{acc, widen(lhs, i16), widen(rhs, i16)});
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16: {
    return arith::AddIOp::create(
        builder, loc, acc,
        call("llvm.x86.avx512.pmaddw.d.512", accType, ValueRange{lhs, rhs}));
  }
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16: {
    return arith::AddIOp::create(
        builder, loc, acc,
        call("llvm.x86.avx512.pmaddw.d.512", accType,
             ValueRange{widen(lhs, i16), widen(rhs, i16)}));
  }
  default:
    return {};
  }
}

/// Lowers a `DataTiledMMAAttr` whose intrinsic is one of the
/// `MMA_GENERIC_SCALAR_1x1x1_REG*` cases directly to a single
/// `vector.contract`. Since the intrinsic's tile shape is 1×1×1, the
/// operand tiles after applying `intrinsics_m` / `intrinsics_n` /
/// `intrinsics_k` are simple row-major (M, K)/(N, K)/(M, N) matmul
/// tiles — no swizzle-based distribution is needed.
static LogicalResult lowerGenericScalarToVectorContract(
    OpBuilder &builder, Location loc, IREE::CPU::DataTiledMMAAttr attr,
    ValueRange inputs, ValueRange outputs, SmallVectorImpl<Value> &results) {
  assert(isGenericScalar(attr.getIntrinsic()) &&
         "lowerGenericScalarToVectorContract only handles "
         "MMA_GENERIC_SCALAR_1x1x1_REG* intrinsics");
  Value lhs = inputs[0];
  Value rhs = inputs[1];
  Value acc = outputs[0];
  Type accElem = cast<VectorType>(acc.getType()).getElementType();
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
  results.push_back(vector::ContractionOp::create(
      builder, loc, lhs, rhs, acc, MapList{{m, k}, {n, k}, {m, n}},
      ArrayRef<vector::IteratorType>{par, par, red}));
  return success();
}

LogicalResult DataTiledMMAAttr::buildUnderlyingOperations(
    OpBuilder &builder, Location loc, ValueRange inputs, ValueRange outputs,
    SmallVectorImpl<Value> &results) const {
  // TODO: handle `transposed_intrinsic`. When set, LHS and RHS swap roles
  // (narrow-N case), and the broadcast in `createCpuMmaIntrinsicCall` would
  // need to broadcast whichever operand is narrower rather than always LHS.
  // Bail for now so we don't silently produce wrong code.
  if (getTransposedIntrinsic()) {
    return failure();
  }

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
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREECPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::CPU
