// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===- ClOptionUtils.h - Utilities for controlling SPIR-V CodeGen ---------===//
//
// This file contains helper functions to read command-line options controlling
// the SPIR-V code generation pipeline. This allows us to put all command-line
// options in one place. Otherwise, we may need to duplicate the same option
// three times: in the pass itself, in the pass pipeline, and in the callers
// constructing the pass pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_CODEGENOPTIONUTILS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_CODEGENOPTIONUTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace iree_compiler {

// Options that can be used to configure SPIR-V code generation.
struct SPIRVCodegenOptions {
  llvm::SmallVector<unsigned, 3> workgroupSize = {};
  llvm::SmallVector<unsigned, 3> workgroupTileSizes = {};
  llvm::SmallVector<unsigned, 3> invocationTileSizes = {};

  bool useWorkgroupMemory = false;
};

// Returns SPIR-V CodeGen options from command-line options.
SPIRVCodegenOptions getSPIRVCodegenOptionsFromClOptions();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_CODEGENOPTIONUTILS_H_
