// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLKernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLSELECTLOWERINGSTRATEGYPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {
/// Selects a strategy for lowering an IREE hal.executable.variant to ROCDL.
class ROCDLSelectLoweringStrategyPass final
    : public impl::ROCDLSelectLoweringStrategyPassBase<
          ROCDLSelectLoweringStrategyPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, IREE::GPU::IREEGPUDialect,
                bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      if (failed(initROCDLLaunchConfig(funcOp))) {
        funcOp.emitOpError("failed to set configuration");
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
