// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUEnums.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::CPU {

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

struct CPUOpaqueMmaLayout {
  int64_t mSize = 0;
  int64_t nSize = 0;
  int64_t kSize = 0;
  Type aType;
  Type bType;
  Type cType;
};

static std::tuple<int64_t, int64_t, int64_t>
getMNKShapeFromIntrinsic(MMAIntrinsic intrinsic) {
  switch (intrinsic) {
  case MMAIntrinsic::None:
    return {0, 0, 0};
  case MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64:
    return {1, 8, 1};
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
    return {1, 16, 1};
  case MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16:
    return {1, 32, 1};
  case MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
    return {1, 16, 2};
  default:
    return {0, 0, 0};
  }
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
  default:
    return {Type(), Type(), Type()};
  }
}

static CPUOpaqueMmaLayout getOpaqueMMALayout(MLIRContext *context,
                                             MMAIntrinsic intrinsic) {
  CPUOpaqueMmaLayout o;
  std::tie(o.aType, o.bType, o.cType) = getABCElementTypes(context, intrinsic);
  std::tie(o.mSize, o.nSize, o.kSize) = getMNKShapeFromIntrinsic(intrinsic);
  return o;
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
  CPUOpaqueMmaLayout o = getOpaqueMMALayout(ctx, intrinsic);
  int64_t m = o.mSize * getIntrinsicsM();
  int64_t n = o.nSize * getIntrinsicsN();
  int64_t k = o.kSize * getIntrinsicsK();
  result.assign({VectorType::get({m, k}, o.aType),
                 VectorType::get({k, n}, o.bType),
                 VectorType::get({m, n}, o.cType)});
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

LogicalResult DataTiledMMAAttr::buildUnderlyingOperations(
    OpBuilder &builder, Location loc, ValueRange inputs, ValueRange outputs,
    SmallVectorImpl<Value> &results) const {
  return failure();
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
