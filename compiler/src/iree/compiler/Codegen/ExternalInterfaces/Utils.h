// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_
#define IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::iree_compiler::IREE {

static const char kEncodingInfoAttrName[] = "encoding_info";

Value calculateStorageSizeInBytesImpl(Attribute attr, Location loc,
                                      OpBuilder &builder, RankedTensorType type,
                                      ValueRange dynamicDims);

/// Returns a dictionary attribute that contains the materialized encoding info,
/// i.e., serialized MaterializeEncodingInfo struct.
/// Requirement: `attr` must implement IREE::Codegen::LayoutAttrInterface.
DictionaryAttr getLayoutImpl(Attribute attr, RankedTensorType type);

} // namespace mlir::iree_compiler::IREE

#endif // IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILSS_H_
