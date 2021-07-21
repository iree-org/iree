// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace iree_compiler {

SPIRVCodegenOptions SPIRVCodegenOptions::getFromCLOptions() {
  static llvm::cl::list<unsigned> clWorkgroupTileSizes(
      "iree-spirv-workgroup-tile-size",
      llvm::cl::desc("Set tile sizes to use for each workgroup when tiling "
                     "Linalg ops in SPIR-V code generation"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

  static llvm::cl::list<unsigned> clInvocationTileSizes(
      "iree-spirv-invocation-tile-size",
      llvm::cl::desc("Set tile sizes for each invocation when tiling Linalg "
                     "ops in SPIR-V code generation"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

  static llvm::cl::opt<bool> clUseWorkgroupMemory(
      "iree-spirv-use-workgroup-memory",
      llvm::cl::desc("Use workgroup memory in SPIR-V code generation"),
      llvm::cl::init(false));

  static llvm::cl::list<unsigned> clWorkgroupSizes(
      "iree-spirv-workgroup-size",
      llvm::cl::desc("Set workgroup size to use for SPIR-V code generation"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

  SPIRVCodegenOptions options;
  options.workgroupSize.assign(clWorkgroupSizes.begin(),
                               clWorkgroupSizes.end());
  options.workgroupTileSizes.assign(clWorkgroupTileSizes.begin(),
                                    clWorkgroupTileSizes.end());
  options.invocationTileSizes.assign(clInvocationTileSizes.begin(),
                                     clInvocationTileSizes.end());
  options.useWorkgroupMemory = clUseWorkgroupMemory;
  return options;
}

}  // namespace iree_compiler
}  // namespace mlir
