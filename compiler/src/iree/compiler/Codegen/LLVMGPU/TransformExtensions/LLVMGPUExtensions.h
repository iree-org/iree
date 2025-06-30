// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS_H_

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {

class DialectRegistry;

namespace scf {
class ForallOp;
class IfOp;
class ForOp;
} // namespace scf

namespace vector {
class VectorDialect;
class WarpExecuteOnLane0Op;
} // namespace vector

} // namespace mlir

namespace mlir::iree_compiler {

class TransformVectorLayoutOptions : public VectorLayoutOptions {
public:
  TransformVectorLayoutOptions(Operation *root, bool fullConversion)
      : VectorLayoutOptions(root, fullConversion) {}

  VectorLayoutInterface getDefaultLayout(VectorType type) const override {
    return VectorLayoutInterface();
  }
};

/// Registers Flow transformations that require IREE-specific information into
/// the transform dialect.
void registerTransformDialectLLVMGPUExtension(DialectRegistry &registry);

namespace IREE::transform_dialect {
// Hook to register LLVMGPU transformations to the transform dialect.
class LLVMGPUExtensions
    : public transform::TransformDialectExtension<LLVMGPUExtensions> {
public:
  LLVMGPUExtensions();
};
} // namespace IREE::transform_dialect

/// Transformation to convert scf.forall to gpu distribution.
FailureOr<SmallVector<OpFoldResult>>
rewriteForallToGpu(scf::ForallOp forallOp,
                   const SmallVector<int64_t> &globalWorkgroupSizes,
                   RewriterBase &rewriter, bool syncAfterDistribute = true);

} // namespace mlir::iree_compiler

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.h.inc"

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS_H_
