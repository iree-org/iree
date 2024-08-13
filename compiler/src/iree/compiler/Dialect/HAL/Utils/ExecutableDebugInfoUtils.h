// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_EXECUTABLEDEBUGINFOUTILS_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_EXECUTABLEDEBUGINFOUTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"

namespace mlir::iree_compiler::IREE::HAL {

// Creates a `[iree.hal.debug.SourceFileDef]` vector from the given sources
// dictionary (filename keys to resource elements contents).
//
// |debugLevel| generally corresponds to the gcc-style levels 0-3:
//   0: no debug information
//   1: minimal debug information
//   2: default debug information
//   3: maximal debug information
flatbuffers_vec_ref_t createSourceFilesVec(int debugLevel,
                                           DictionaryAttr sourcesAttr,
                                           FlatbufferBuilder &fbb);

// Creates one `iree.hal.debug.ExportDef` for every export and returns them in
// the same order.
//
// |debugLevel| generally corresponds to the gcc-style levels 0-3:
//   0: no debug information
//   1: minimal debug information
//   2: default debug information
//   3: maximal debug information
SmallVector<flatbuffers_ref_t>
createExportDefs(int debugLevel,
                 ArrayRef<IREE::HAL::ExecutableExportOp> exportOps,
                 FlatbufferBuilder &fbb);

} // namespace mlir::iree_compiler::IREE::HAL

#endif //  IREE_COMPILER_DIALECT_HAL_UTILS_EXECUTABLEDEBUGINFOUTILS_H_
