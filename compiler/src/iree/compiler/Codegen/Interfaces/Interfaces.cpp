// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/Interfaces.h"

#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h"
// TODO: Remove this dependency once the transform dialect extensions
// have a better registration mechanism.
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"

namespace mlir {
namespace iree_compiler {

void registerCodegenInterfaces(DialectRegistry &registry) {
  registerProcessorOpInterfaceExternalModels(registry);
  registerBufferizationInterfaces(registry);
  // TODO: Remove this dependency once the transform dialect extensions
  // have a better registration mechanism.
  // TODO: when warranted, move to its own file.
  registry.addExtensions<IREE::LinalgExt::LinalgExtTransformOpsExtension,
                         transform_ext::StructuredTransformOpsExtension>();
  registerPartitionableLoopsInterfaceModels(registry);
  registerTransformDialectCommonExtension(registry);
  registerTransformDialectFlowExtension(registry);
  registerTransformDialectLLVMCPUExtension(registry);
  registerTransformDialectLLVMGPUExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  linalg::registerTilingInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
}

}  // namespace iree_compiler
}  // namespace mlir
