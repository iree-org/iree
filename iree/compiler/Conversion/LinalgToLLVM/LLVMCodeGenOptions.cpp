// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/LinalgToLLVM/LLVMCodeGenOptions.h"

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<bool> clConvImg2ColConversion(
    "iree-codegen-linalg-to-llvm-conv-img2col-conversion",
    llvm::cl::desc("Enable rewriting linalg.conv_2d_input_nhwc_filter_hwcf "
                   "linalg.generic that does img2col buffer packing + "
                   "linag.matmul"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clUnfusedFMA(
    "iree-codegen-linalg-to-llvm-use-unfused-fma",
    llvm::cl::desc("Enable rewriting llvm.fma to its unfused version."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableLinalgOnTensorsToVectors(
    "iree-codegen-linalg-to-llvm-linalg-on-tensors-to-vectors",
    llvm::cl::desc("Enable rewriting llvm.fma to its unfused version."),
    llvm::cl::init(false));

LLVMCodegenOptions getLLVMCodegenOptionsFromClOptions() {
  LLVMCodegenOptions options;
  options.useConvImg2Col = clConvImg2ColConversion;
  options.unfuseFMAOps = clUnfusedFMA;
  options.useLinalgOnTensorsToVectors = clEnableLinalgOnTensorsToVectors;
  return options;
}

}  // namespace iree_compiler
}  // namespace mlir
