// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-cast-address-space-function"

namespace mlir::iree_compiler {

namespace {

struct LLVMGPUCastAddressSpaceFunctionPass
    : public LLVMGPUCastAddressSpaceFunctionBase<
          LLVMGPUCastAddressSpaceFunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    IRRewriter rewriter(context);

    ModuleOp moduleOp = getOperation();

    auto castOperands = [&](mlir::Operation::operand_range operands,
                            SmallVector<Value> &newOperands) {
      bool anyCasted = false;
      for (auto operand : operands) {
        if (auto memrefType = dyn_cast<mlir::MemRefType>(operand.getType())) {
          if (hasSharedMemoryAddressSpace(memrefType)) {
            mlir::MemRefType new_memrefType = mlir::MemRefType::get(
                memrefType.getShape(), memrefType.getElementType(),
                memrefType.getLayout());
            operand = rewriter.create<memref::MemorySpaceCastOp>(
                operand.getLoc(), new_memrefType, operand);
            anyCasted = true;
          }
        }
        newOperands.push_back(operand);
      }
      return anyCasted;
    };

    moduleOp->walk([&](mlir::CallOpInterface callOp) {
      auto callee = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
      SmallVector<Value> newOperands;
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(callOp);
      if (castOperands(callOp->getOperands(), newOperands)) {
        callOp.getArgOperandsMutable().assign(newOperands);
        auto fnDecl = dyn_cast_or_null<mlir::FunctionOpInterface>(
            SymbolTable::lookupSymbolIn(moduleOp, callee));
        if (fnDecl) {
          SmallVector<Type> callArgumentTypes;
          for (auto op : newOperands)
            callArgumentTypes.push_back(op.getType());
          FunctionType functionType = rewriter.getFunctionType(
              callArgumentTypes, fnDecl->getResultTypes());
          fnDecl.setType(functionType);
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMGPUCastAddressSpaceFunction() {
  return std::make_unique<LLVMGPUCastAddressSpaceFunctionPass>();
}

} // namespace mlir::iree_compiler
