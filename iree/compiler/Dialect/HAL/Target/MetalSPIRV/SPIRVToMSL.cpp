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

#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/SPIRVToMSL.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "spirv_cross/spirv_msl.hpp"

#define DEBUG_TYPE "spirv-to-msl"

namespace mlir {
namespace iree_compiler {

using SPIRVToMSLCompiler = SPIRV_CROSS_NAMESPACE::CompilerMSL;

std::string crossCompileSPIRVToMSL(llvm::ArrayRef<uint32_t> spvBinary,
                                   const std::string &entryPoint) {
  SPIRVToMSLCompiler spvCrossCompiler(spvBinary.data(), spvBinary.size());

  // All spirv-cross operations work on the current entry point. It should be
  // set right after the cross compiler construction.
  spvCrossCompiler.set_entry_point(
      entryPoint, spv::ExecutionModel::ExecutionModelGLCompute);

  // TODO(antiagainst): fill out the following according to the Metal GPU
  // family.
  SPIRVToMSLCompiler::Options spvCrossOptions;
  spvCrossOptions.platform = SPIRVToMSLCompiler::Options::Platform::macOS;
  spvCrossOptions.msl_version =
      SPIRVToMSLCompiler::Options::make_msl_version(2, 0);
  // Eanble using Metal argument buffers. It is more akin to Vulkan descriptor
  // sets, which is how IREE HAL models resource bindings and mappings.
  spvCrossOptions.argument_buffers = true;
  spvCrossCompiler.set_msl_options(spvCrossOptions);

  std::string mslSource = spvCrossCompiler.compile();
  LLVM_DEBUG(llvm::dbgs()
             << "Cross compiled Metal Shading Language source code:\n-----\n"
             << mslSource << "\n-----\n");

  return mslSource;
}

}  // namespace iree_compiler
}  // namespace mlir
