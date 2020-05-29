// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

#include "mlir/Dialect/SPIRV/SPIRVLowering.h"

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
