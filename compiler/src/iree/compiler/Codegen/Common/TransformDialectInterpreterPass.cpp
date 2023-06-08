// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for codegen.
/// This needs to be its own pass because the registration mechanism and ops
/// available are different than for other interpreters.
class TransformDialectInterpreterPass
    : public mlir::transform::TransformInterpreterPassBase<
          TransformDialectInterpreterPass,
          iree_compiler::TransformDialectInterpreterBase> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: this is only necessary to make registry subset happy when running
    // the lowering to LLVM. The lowering should be changed to stop using the
    // nested pass manager and this will go away.

    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    mlir::iree_compiler::IREE::Flow::FlowDialect,
                    arith::ArithDialect,
                   affine::AffineDialect,
                    bufferization::BufferizationDialect,
                    func::FuncDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    linalg::transform::LinalgTransformDialect,
                    LLVM::LLVMDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect,
                    vector::VectorDialect
        // clang-format on
        >();

    // TODO: these should be registered by the extension instead, but there is
    // no support for it in core currently.
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerFindPayloadReplacementOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);

    registry.addExtensions<
        mlir::iree_compiler::IREE::LinalgExt::LinalgExtTransformOpsExtension,
        transform_ext::StructuredTransformOpsExtension>();
    iree_compiler::registerTransformDialectCommonExtension(registry);
    iree_compiler::registerTransformDialectFlowExtension(registry);
    iree_compiler::registerTransformDialectLLVMCPUExtension(registry);
    iree_compiler::registerTransformDialectLLVMGPUExtension(registry);
    affine::registerTransformDialectExtension(registry);
    bufferization::registerTransformDialectExtension(registry);
    gpu::registerTransformDialectExtension(registry);
    linalg::registerTransformDialectExtension(registry);
    memref::registerTransformDialectExtension(registry);
    scf::registerTransformDialectExtension(registry);
    vector::registerTransformDialectExtension(registry);
  }

  TransformDialectInterpreterPass(
      StringRef transformFileName = StringRef(),
      StringRef debugPayloadRootTag = StringRef(),
      StringRef debugTransformRootTag = StringRef()) {
    this->transformFileName = transformFileName.str();
    this->debugPayloadRootTag = debugPayloadRootTag.str();
    this->debugTransformRootTag = debugTransformRootTag.str();
  }
  TransformDialectInterpreterPass(const TransformDialectInterpreterPass &pass) =
      default;
};
}  // namespace

namespace mlir {
namespace iree_compiler {
/// Create a Transform dialect interpreter pass.
std::unique_ptr<Pass> createTransformDialectInterpreterPass(
    llvm::StringRef transformFileName, llvm::StringRef debugPayloadRootTag,
    llvm::StringRef debugTransformRootTag) {
  return std::make_unique<TransformDialectInterpreterPass>(
      transformFileName, debugPayloadRootTag, debugTransformRootTag);
}
}  // namespace iree_compiler
}  // namespace mlir
