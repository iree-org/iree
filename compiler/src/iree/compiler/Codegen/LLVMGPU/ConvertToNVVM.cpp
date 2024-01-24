// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace {

int kDefaultCUDACapability = 80;

/// Return the CUDA capability of the gpu. Assumes CUDA capability is 80 (sm_80)
/// if not specified.
static int getCUDACapbility(Operation *op) {
  FailureOr<IREE::HAL::ExecutableVariantOp> variantOp =
      getExecutableVariantOp(op);
  if (failed(variantOp)) {
    return kDefaultCUDACapability;
  }

  auto targetAttr = variantOp->getTargetAttr();
  if (auto config = targetAttr.getConfiguration()) {
    if (auto attr = config.getAs<StringAttr>("target_arch")) {
      StringRef targetName = attr.getValue();
      APInt version;
      if (targetName.starts_with("sm_") &&
          !targetName.substr(3).getAsInteger(10, version)) {
        return version.getZExtValue();
      }
    }
  }
  return kDefaultCUDACapability;
}

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct ConvertToNVVMPass : public ConvertToNVVMBase<ConvertToNVVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(m.getContext(), DataLayout(m));
    options.overrideIndexBitwidth(64);
    LLVMTypeConverter converter(m.getContext(), options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kGlobalMemorySpace);
          case gpu::AddressSpace::Workgroup:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kSharedMemorySpace);
          case gpu::AddressSpace::Private:
            return 0;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
    // Lowering for MMAMatrixType.
    converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToLLVMType(type);
    });
    // Convert dummy tokens.
    converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) -> Type {
      return converter.convertType(IntegerType::get(type.getContext(), 32));
    });
    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    // Run Vector -> Vector transformations ahead of conversion to LLVM.
    {
      RewritePatternSet patterns(&getContext());
      populateDropSharedMemoryDeallocOpPatterns(patterns);
      populateScalarizeMathOps(patterns);
      populateConvertSharedMemoryAllocOps(patterns);
      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      // We currently always use 64 bit indices, thus ensure the bit width of
      // the mask compare is consistent.
      vector::populateVectorMaskMaterializationPatterns(
          patterns, /*force32BitVectorIndices=*/false);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      // TODO: doubtful that the "default" does what one want here, it is likely
      // better to use something else.
      vector::populateVectorTransposeLoweringPatterns(
          patterns, vector::VectorTransformsOptions());
      vector::populateVectorTransferLoweringPatterns(patterns);
      arith::populateExpandBFloat16Patterns(patterns);
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
      // Convert arith::maximumf/minimumf ops on older gpus since the lowering
      // is faulty for them.
      // TODO: Remove this once the lowering in LLVM is fixed
      // (https://github.com/llvm/llvm-project/issues/64606).
      if (getCUDACapbility(m) < 80) {
        RewritePatternSet patterns(&getContext());
        populateReplaceSlowMinMaxOpsPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
          return signalPassFailure();
        }
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
      populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
      populateNVGPUToNVVMConversionPatterns(converter, llvmPatterns);
      populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
      LLVMConversionTarget target(getContext());
      configureGpuToNVVMConversionLegality(target);
      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns)))) {
        signalPassFailure();
      }
    }
    // Convert NVVM ops to Inline Assembly.
    {
      RewritePatternSet patterns(&getContext());
      populateNVVMToLLVMConversionPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    ConvertToDynamicSharedMemory(m);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass() {
  return std::make_unique<ConvertToNVVMPass>();
}

} // namespace mlir::iree_compiler
