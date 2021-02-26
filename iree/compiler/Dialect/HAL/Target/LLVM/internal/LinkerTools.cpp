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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(benvanik): add other platforms:
// createMacLinkerTool using ld64.lld
// createWasmLinkerTool wasm-ld

std::unique_ptr<LinkerTool> createAndroidLinkerTool(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions);
std::unique_ptr<LinkerTool> createUnixLinkerTool(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions);
std::unique_ptr<LinkerTool> createWindowsLinkerTool(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions);

// static
std::unique_ptr<LinkerTool> LinkerTool::getForTarget(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  if (targetTriple.isAndroid()) {
    return createAndroidLinkerTool(targetTriple, targetOptions);
  } else if (targetTriple.isOSWindows() ||
             targetTriple.isWindowsMSVCEnvironment()) {
    return createWindowsLinkerTool(targetTriple, targetOptions);
  }
  return createUnixLinkerTool(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
