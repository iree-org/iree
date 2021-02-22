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

#include "iree/compiler/Conversion/LinalgToSPIRV/CodeGenOptionUtils.h"

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace iree_compiler {

SPIRVCodegenOptions getSPIRVCodegenOptionsFromClOptions() {
  static llvm::cl::opt<bool> clEnableVectorization(
      "iree-spirv-enable-vectorization",
      llvm::cl::desc(
          "Enable vectorization transformations in SPIR-V code generation"),
      llvm::cl::init(false));

  static llvm::cl::list<unsigned> clTileSizes(
      "iree-spirv-tile-size",
      llvm::cl::desc("Set tile sizes to use for tiling Linalg operations in "
                     "SPIR-V code generation"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

  static llvm::cl::opt<bool> clUseWorkgroupMemory(
      "iree-spirv-use-workgroup-memory",
      llvm::cl::desc("Use workgroup memory in SPIR-V code generation"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> clVectorizeMemref(
      "iree-spirv-enable-memref-vectorization",
      llvm::cl::desc("Vectorize memref if possible in SPIR-V code generation"),
      llvm::cl::init(false));

  static llvm::cl::list<unsigned> clWorkgroupSizes(
      "iree-spirv-workgroup-size",
      llvm::cl::desc("Set workgroup size to use for SPIR-V code generation"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

  static llvm::cl::opt<bool> clEnableLinalgOnTensorsSPIRV(
      "iree-codegen-spirv-experimental-linalg-on-tensors",
      llvm::cl::desc("Enable the linalg on tensors on SPIR-V path"),
      llvm::cl::init(false));

  SPIRVCodegenOptions options;
  options.workgroupSize.assign(clWorkgroupSizes.begin(),
                               clWorkgroupSizes.end());
  options.tileSizes.assign(clTileSizes.begin(), clTileSizes.end());
  options.enableVectorization = clEnableVectorization;
  options.useWorkgroupMemory = clUseWorkgroupMemory;
  options.vectorizeMemref = clVectorizeMemref;
  options.usingLinalgOnTensors = clEnableLinalgOnTensorsSPIRV;
  return options;
}

}  // namespace iree_compiler
}  // namespace mlir
