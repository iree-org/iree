// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"

namespace mlir::iree_compiler::IREE::GPU {

static SmallVector<int64_t> getIntegerVector(ArrayAttr array) {
  if (!array || !llvm::all_of(array.getValue(), llvm::IsaPred<IntegerAttr>)) {
    return {};
  }
  return llvm::map_to_vector(array.getValue(), [](Attribute s) -> int64_t {
    return cast<IntegerAttr>(s).getInt();
  });
}

constexpr StringLiteral kMmaKindName = "mma_kind";

IREE::GPU::MmaInterfaceAttr getMmaKind(LoweringConfigAttr config) {
  return config.getAttributes().getAs<IREE::GPU::MmaInterfaceAttr>(
      kMmaKindName);
}

void setMmaKind(MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
                IREE::GPU::MmaInterfaceAttr kind) {
  attrs.emplace_back(StringAttr::get(context, kMmaKindName), kind);
}

// TODO: Merge subgroup counts functionality into subgroup tiling level
//       lowering, when we have it implemented.
constexpr StringLiteral kSubgroupMCountName = "subgroup_m_count";
constexpr StringLiteral kSubgroupNCountName = "subgroup_n_count";

std::optional<int64_t> getSubgroupMCount(LoweringConfigAttr config) {
  auto subgroup_m_count_attr =
      config.getAttributes().getAs<IntegerAttr>(kSubgroupMCountName);
  if (!subgroup_m_count_attr) {
    return std::nullopt;
  }
  return subgroup_m_count_attr.getInt();
}

std::optional<int64_t> getSubgroupNCount(LoweringConfigAttr config) {
  auto subgroup_n_count_attr =
      config.getAttributes().getAs<IntegerAttr>(kSubgroupNCountName);
  if (!subgroup_n_count_attr) {
    return std::nullopt;
  }
  return subgroup_n_count_attr.getInt();
}

void setSubgroupMCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroup_m_count) {
  attrs.emplace_back(
      StringAttr::get(context, kSubgroupMCountName),
      IntegerAttr::get(IntegerType::get(context, 64), subgroup_m_count));
}

void setSubgroupNCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroup_n_count) {
  attrs.emplace_back(
      StringAttr::get(context, kSubgroupNCountName),
      IntegerAttr::get(IntegerType::get(context, 64), subgroup_n_count));
}

constexpr StringLiteral kPromoteOperandsName = "promote_operands";

std::optional<SmallVector<int64_t>>
getPromotedOperandList(LoweringConfigAttr config) {
  auto array = config.getAttributes().getAs<ArrayAttr>(kPromoteOperandsName);
  if (!array) {
    return std::nullopt;
  }
  return getIntegerVector(array);
}

void setPromotedOperandList(MLIRContext *context,
                            SmallVectorImpl<NamedAttribute> &attrs,
                            ArrayRef<int64_t> operands) {
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, kPromoteOperandsName),
                     b.getI64ArrayAttr(operands));
}

constexpr StringLiteral kPaddingName = "padding";

std::optional<SmallVector<int64_t>> getPaddingList(LoweringConfigAttr config) {
  auto array = config.getAttributes().getAs<ArrayAttr>(kPaddingName);
  if (!array) {
    return std::nullopt;
  }
  return getIntegerVector(array);
}

} // namespace mlir::iree_compiler::IREE::GPU
