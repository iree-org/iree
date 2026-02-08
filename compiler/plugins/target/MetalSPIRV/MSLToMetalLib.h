// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_TARGET_METALSPIRV_MSLTOMETALLIB_H_
#define IREE_COMPILER_PLUGINS_TARGET_METALSPIRV_MSLTOMETALLIB_H_

#include <optional>
#include <string>

#include "compiler/plugins/target/MetalSPIRV/MetalTargetPlatform.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir::iree_compiler::IREE::HAL {

// Finds the Metal offline toolchain (xcrun with metal and metallib tools).
// Returns the resolved xcrun path, or std::nullopt if tools are unavailable.
std::optional<std::string> findMetalToolchain();

// Compiles |mslCode| into a Metal library binary using the toolchain at
// |xcrunPath| (from findMetalToolchain). On failure, returns nullptr and
// populates |errMsg| with compiler diagnostics.
std::unique_ptr<llvm::MemoryBuffer>
compileMSLToMetalLib(MetalTargetPlatform targetPlatform,
                     llvm::StringRef mslCode, llvm::StringRef entryPoint,
                     llvm::StringRef xcrunPath, std::string &errMsg);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_PLUGINS_TARGET_METALSPIRV_MSLTOMETALLIB_H_
