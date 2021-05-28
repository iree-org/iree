// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MemorySpace.h - Methods for codifying memory space values ----------===//
//
// Method that codify the convention for memory space values. MLIR itself does
// not have definition for the memory space values on (memref) types. It is left
// to the clients to do so. Here this is codified for IREE codegen. It is made
// consistent with what the lowering from Standard to SPIR-V in MLIR expects.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_MEMORYSPACE_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_MEMORYSPACE_H_

#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace mlir {
namespace iree_compiler {

/// Returns the memref memory space to use with memrefs for workgroup memory.
inline unsigned getWorkgroupMemorySpace() {
  return SPIRVTypeConverter::getMemorySpaceForStorageClass(
      spirv::StorageClass::Workgroup);
}
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_MEMORYSPACE_H_
