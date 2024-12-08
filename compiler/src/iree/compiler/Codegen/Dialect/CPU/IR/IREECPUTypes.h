// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "llvm/TargetParser/Triple.h"

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h.inc"
#undef GET_ATTRDEF_CLASSES
// clang-format on

namespace mlir::iree_compiler::IREE::CPU {

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

/// Returns the best TileMxNxK from `enumeratedTiles` pool. If the
/// `hostDefinedUpperBound` is not empty, the chosen tile sizes can not be
/// greater than the values.
/// TODO(#16933): Remove `hostDefinedUpperBound` once we can propagate such
/// information to host. For now, they are defined by host.
Codegen::TileMxNxK
chooseMatmulTile(ArrayRef<Codegen::TileMxNxK> enumeratedTiles,
                 IREE::Encoding::MatmulNarrowDim narrowDim,
                 ArrayRef<int64_t> hostDefinedUpperBound = {});

/// Returns the StringAttr with the name `str` in the `config`, if found.
std::optional<StringAttr> getConfigStringAttr(DictionaryAttr config,
                                              StringRef str);

/// Returns the LLVM Target triple associated with the `config`, if set.
std::optional<llvm::Triple> getTargetTriple(DictionaryAttr config);

/// Returns true if `config` has `feature` in its CPU features.
bool hasFeature(DictionaryAttr config, StringRef feature);

// Architecture identification.
bool isX86(DictionaryAttr config);
bool isX86_64(DictionaryAttr config);
bool isAArch64(DictionaryAttr config);
bool isRISCV(DictionaryAttr config);
bool isRISCV32(DictionaryAttr config);

} // namespace mlir::iree_compiler::IREE::CPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_
