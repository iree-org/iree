// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding ROCDL equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct ConvertToROCDLPass : public ConvertToROCDLBase<ConvertToROCDLPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(m.getContext(), DataLayout(m));
    options.overrideIndexBitwidth(64);
    LLVMTypeConverter converter(m.getContext(), options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) {
          switch (space) {
            case gpu::AddressSpace::Global:
              return 1;
            case gpu::AddressSpace::Workgroup:
              return 3;
            case gpu::AddressSpace::Private:
              return 5;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    // Run Vector -> Vector transformations ahead of conversion to LLVM.
    {
      RewritePatternSet patterns(&getContext());
      populateScalarizeMathOps(patterns);
      populateConvertSharedMemoryAllocOps(patterns);
      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      // TODO: doubtful that the "default" does what one want here, it is likely
      // better to use something else.
      vector::populateVectorTransposeLoweringPatterns(
          patterns, vector::VectorTransformsOptions());
      vector::populateVectorTransferLoweringPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      populateGpuRewritePatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    {
      RewritePatternSet llvmPatterns(&getContext());
      populateLowerHALInterfaceOp(llvmPatterns);
      populateLLVMConversionPatterns(&getContext(), llvmPatterns, converter);
      populateComplexToLLVMConversionPatterns(converter, llvmPatterns);
      populateMathToLLVMConversionPatterns(converter, llvmPatterns);
      memref::populateExpandStridedMetadataPatterns(llvmPatterns);
      populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
      populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
      cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
      arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
      populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
      populateGpuToROCDLConversionPatterns(converter, llvmPatterns,
                                           gpu::amd::Runtime::Unknown);
      LLVMConversionTarget target(getContext());
      populateFuncToLLVMFuncOpConversionPattern(converter, llvmPatterns);
      configureGpuToROCDLConversionLegality(target);
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp funcOp) {
        if (isEntryPoint(funcOp)) return false;
        return true;
      });
      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
        signalPassFailure();
    }
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertToROCDLPass() {
  return std::make_unique<ConvertToROCDLPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
