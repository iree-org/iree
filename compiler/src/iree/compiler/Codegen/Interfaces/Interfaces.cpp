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
#include "iree/compiler/Dialect/LinalgExt/TransformExtensions/LinalgExtExtensionsOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"

namespace mlir::iree_compiler {

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
  linalg::registerTilingInterfaceExternalModels(registry);
  affine::registerTransformDialectExtension(registry);
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  bufferization::registerTransformDialectExtension(registry);
  gpu::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  memref::registerTransformDialectExtension(registry);
  memref::registerValueBoundsOpInterfaceExternalModels(registry);
  scf::registerTransformDialectExtension(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerTransformDialectExtension(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  transform::registerLoopExtension(registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);
  vector::registerTransformDialectExtension(registry);
}

} // namespace mlir::iree_compiler
