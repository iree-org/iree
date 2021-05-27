// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_LINALGTOLLVM_LLVMCODEGENOPTIONS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOLLVM_LLVMCODEGENOPTIONS_H_

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace iree_compiler {

// Options used to configure LLVM passes.
struct LLVMCodegenOptions {
  bool useConvImg2Col = false;
  // Target specific options.
  bool unfuseFMAOps = false;
  bool useVectorToAarch64 = false;
  bool useLinalgOnTensorsToVectors = false;
};

// Returns LLVM CodeGen options from command-line options.
LLVMCodegenOptions getLLVMCodegenOptionsFromClOptions();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOLLVM_LLVMCODEGENOPTIONS_H_
