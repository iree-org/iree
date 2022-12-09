// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <type_traits>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterUtils.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-transform-dialect-jitter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
namespace {
/// Pass declaration.
/// Jitter pass that applies transform dialect ops for codegen.
/// This needs to be its own pass because the registration mechanism and ops
/// available are different than for other jitters.
class TransformDialectJitterPass
    : public iree_compiler::TransformDialectJitterBase<
          TransformDialectJitterPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: this is only necessary to make registry subset happy when
    // running the lowering to LLVM. The lowering should be changed to stop
    // using the nested pass manager and this will go away.

    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    mlir::iree_compiler::IREE::Flow::FlowDialect,
                    arith::ArithDialect,
                    AffineDialect,
                    bufferization::BufferizationDialect,
                    BuiltinDialect,
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
                    vector::VectorDialect>();
    // clang-format on

    // TODO: these should be registered by the extension instead, but there
    // is no support for it in core currently.
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);

    registry.addExtensions<
        mlir::iree_compiler::IREE::LinalgExt::LinalgExtTransformOpsExtension,
        transform_ext::StructuredTransformOpsExtension>();
    iree_compiler::registerTransformDialectCommonExtension(registry);
    iree_compiler::registerTransformDialectFlowExtension(registry);
    iree_compiler::registerTransformDialectLLVMCPUExtension(registry);
    iree_compiler::registerTransformDialectLLVMGPUExtension(registry);
    linalg::registerTransformDialectExtension(registry);
  }

  TransformDialectJitterPass() = default;

  void runOnOperation() override {
    Operation *target = getOperation();
    LLVM_DEBUG(DBGS() << "input IR:\n" << *target);
    Region *topLevelTransformRegion = nullptr;
    target->walk(
        [&](::transform_ext::CanonicalizedSequenceOp transformTopLevelOp) {
          topLevelTransformRegion = transformTopLevelOp->getParentRegion();
        });

    LLVM_DEBUG(DBGS() << "apply transform:\n"
                      << *topLevelTransformRegion->getParentOp());
    if (failed(transform::applyTransformsInRegion(*topLevelTransformRegion,
                                                  target))) {
      target->emitOpError() << "transform dialect jitter failed";
      return signalPassFailure();
    }
  }
};
}  // namespace

namespace mlir {
namespace iree_compiler {
/// Create a Transform dialect jitter pass.
std::unique_ptr<Pass> createTransformDialectJitterPass() {
  return std::make_unique<TransformDialectJitterPass>();
}
}  // namespace iree_compiler
}  // namespace mlir
