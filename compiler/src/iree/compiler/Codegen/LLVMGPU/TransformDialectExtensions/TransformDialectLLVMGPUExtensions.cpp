// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TransformDialectLLVMGPUExtensions.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::TransformDialectLLVMGPUExtensions::
    TransformDialectLLVMGPUExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectLLVMGPUExtension(
    DialectRegistry &registry) {
  registry
      .addExtensions<transform_dialect::TransformDialectLLVMGPUExtensions>();
}

// TODO: Maybe we need both a transform.iree.cpu.bufferize and a
// transform.iree.gpu.bufferize rather than a single common bufferize op?

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.cpp.inc"
