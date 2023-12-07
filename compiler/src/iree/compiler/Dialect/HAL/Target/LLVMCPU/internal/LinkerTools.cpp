// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.h"

namespace mlir::iree_compiler::IREE::HAL {

// TODO(benvanik): add other platforms:
// createMacLinkerTool using ld64.lld

std::unique_ptr<LinkerTool>
createAndroidLinkerTool(const llvm::Triple &targetTriple,
                        LLVMTargetOptions &targetOptions);
std::unique_ptr<LinkerTool>
createEmbeddedLinkerTool(const llvm::Triple &targetTriple,
                         LLVMTargetOptions &targetOptions);
std::unique_ptr<LinkerTool>
createUnixLinkerTool(const llvm::Triple &targetTriple,
                     LLVMTargetOptions &targetOptions);
std::unique_ptr<LinkerTool>
createWasmLinkerTool(const llvm::Triple &targetTriple,
                     LLVMTargetOptions &targetOptions);
std::unique_ptr<LinkerTool>
createWindowsLinkerTool(const llvm::Triple &targetTriple,
                        LLVMTargetOptions &targetOptions);

// static
std::unique_ptr<LinkerTool>
LinkerTool::getForTarget(const llvm::Triple &targetTriple,
                         LLVMTargetOptions &targetOptions) {
  if (targetOptions.target.getLinkEmbedded()) {
    return createEmbeddedLinkerTool(targetTriple, targetOptions);
  } else if (targetTriple.isAndroid()) {
    return createAndroidLinkerTool(targetTriple, targetOptions);
  } else if (targetTriple.isOSWindows() ||
             targetTriple.isWindowsMSVCEnvironment()) {
    return createWindowsLinkerTool(targetTriple, targetOptions);
  } else if (targetTriple.isWasm()) {
    return createWasmLinkerTool(targetTriple, targetOptions);
  }
  return createUnixLinkerTool(targetTriple, targetOptions);
}

} // namespace mlir::iree_compiler::IREE::HAL
