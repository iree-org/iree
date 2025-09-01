// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

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

/// Returns the entry key for the config in IREE::CPU::LoweringConfigAttr.
/// Returns null if `level` is invalid.
StringRef getTilingLevelName(TilingLevel level) {
  switch (level) {
  case DistributionTiles:
    return kDistributionConfigKey;
  case CacheParallelTiles:
    return kCacheParallelConfigKey;
  case CacheReductionTiles:
    return kCacheReductionConfigKey;
  case VectorCommonParallelTiles:
    return kVectorCommonParallelConfigKey;
  case VectorReductionTiles:
    return kVectorReductionConfigKey;
  case VectorInnerParallelTiles:
    return kVectorInnerParallelConfigKey;
  case MaxNumTileLevels:
  case InvalidLevel:
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
  for (unsigned i = 0, e = TilingLevel::MaxNumTileLevels; i < e; ++i) {
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
  return getTileSizes(getConfig(), DistributionTiles);
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
  for (auto level : vectorTilingLevels) {
    if (!hasTilingLevel(level)) {
      continue;
    }
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        getTilingLevelAttr(level));
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
  for (auto level : vectorTilingLevels) {
    if (!hasTilingLevel(level)) {
      continue;
    }
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        getTilingLevelAttr(level));
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
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREECPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::CPU
