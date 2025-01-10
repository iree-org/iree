// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE {
using Codegen::MaterializeEncodingInfo;

DictionaryAttr getLayoutImpl(Attribute attr, RankedTensorType type) {
  MLIRContext *ctx = attr.getContext();
  auto deviceLayoutAttr = cast<IREE::Codegen::LayoutAttrInterface>(attr);
  const MaterializeEncodingInfo info = deviceLayoutAttr.getEncodingInfo(type);
  auto strAttr = StringAttr::get(ctx, "encoding_info");
  Attribute encodingInfoAttr =
      IREE::Codegen::serializeEncodingInfo(attr.getContext(), info);
  return DictionaryAttr::get(ctx, {NamedAttribute(strAttr, encodingInfoAttr)});
}

} // namespace mlir::iree_compiler::IREE
